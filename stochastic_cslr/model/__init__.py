import copy
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from collections import Counter

from .sgs import SGSResNet18
from .xfmr import TransformerEncoder
from .dec import Decoder


class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_num_states,
        dim=512,
        rdim=32,
        p_detach=0.75,
        rpe_k=8,
        use_sfl=True,
        ent_coef=lambda: 1.0,
        heads=4,
        semantic_layers=2,
        dropout=0.1,
        monte_carlo_samples=32,
    ):
        """
        Args:
            vocab_size: vocabulary size of the dataset.
            max_num_states: max number of state per gloss.
            dim: hidden dim for transformer encoder.
            rdim: hidden dim for the state number predictor and baseline.
            p_detach: gradient stopping proportion.
            rpe_k: the window size (one side) for relative postional encoding.
            use_sfl: whether to use stochastic fine-grained labeling.
            ent_coef: entropy loss coefficient, larger the predictor converges slower.
            heads: number of heads for transformer encoder.
            semantic_layers: number of layers for transformer encoder.
            dropout: p_dropout.
            monte_carlo_samples: number of Monte Carlo sampling for stochastic fine-grained labeling.
        """
        super().__init__()
        self.use_sfl = use_sfl
        self.monte_carlo_samples = monte_carlo_samples
        self.ent_coef = ent_coef

        self.max_num_states = max_num_states
        self.vocab_size = vocab_size

        self.visual = SGSResNet18(
            dim,
            p_detach,
        )

        self.semantic = TransformerEncoder(
            dim,
            heads,
            semantic_layers,
            dropout,
            rpe_k,
        )

        self.decoder = Decoder(
            vocab_size,
            max_num_states,
        )

        # plus 1 for blank
        self.classifier = nn.Linear(dim, self.decoder.total_states + 1, bias=False)

        self.blank = self.decoder.total_states  # last dim as blank

        if use_sfl:
            self.predictor = nn.Sequential(
                nn.Embedding(vocab_size, rdim),
                nn.Linear(rdim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, max_num_states),
            )

            self.baseline = nn.Sequential(
                nn.Embedding(vocab_size, rdim),
                nn.Linear(rdim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, rdim),
                nn.ReLU(),
                nn.Linear(rdim, 1),
            )

    def forward(self, x):
        """
        Args:
            x: list of (t c h w)
        Return:
            log probs [(t n)]
        """
        xl = list(map(len, x))
        x = self.visual(x)
        x = self.semantic(x)
        x = torch.cat(x)
        x = self.classifier(x)
        x = x.log_softmax(dim=-1)
        x = x.split(xl)
        return x

    def expand(self, y, n=None):
        """Expand to tensor"""
        return torch.tensor(self.decoder.expand(y, n)).to(y.device)

    def compute_ctc_loss(self, x, y, reduction="mean"):
        """
        Args:
            x: log_probs, (t d)
            y: labels, (t')
        Return:
            loss
        """
        xl = torch.tensor(list(map(len, x)))
        yl = torch.tensor(list(map(len, y)))
        x = pad_sequence(x, False)  # -> (t b c)
        y = pad_sequence(y, True)  # -> (b s)
        return F.ctc_loss(x, y, xl, yl, self.blank, reduction, True)

    @staticmethod
    def mean_over_time(l):
        return torch.stack([li.mean() for li in l]).to(l[0].device)

    @property
    def nsm1_dist(self):
        device = next(self.predictor.parameters()).device
        logits = self.predictor(torch.arange(self.vocab_size).to(device))
        return Categorical(logits=logits)

    @property
    def most_probable_num_states(self):
        return (self.nsm1_dist.probs.argmax(-1) + 1).cpu()

    def compute_sfl_losses(self, lp, y):
        """
        Args:
            lp: log probs, (t c)
            y: label, (t')
        """
        yl = list(map(len, y))
        y = torch.cat(y)

        ylogits = self.predictor(y)
        dist = Categorical(logits=ylogits)

        if self.training:
            nsm1 = dist.sample()  # num states minus 1 (nsm1)
        else:
            nsm1 = dist.probs.argmax(dim=-1)

        nll = -dist.log_prob(nsm1)
        nll = self.mean_over_time(nll.split(yl))

        s = [self.expand(yi, m1i + 1) for yi, m1i in zip(y.split(yl), nsm1.split(yl))]
        sl = list(map(len, s))

        ctc_loss = self.compute_ctc_loss(lp, s, reduction="none")
        ctc_loss = self.mean_over_time([ci / sli for ci, sli in zip(ctc_loss, sl)])

        # some ctc loss is zerored out due to invalid length
        # zero out the corresponding baseline will disable
        # the nll loss and bsl loss at the same time
        baseline = self.mean_over_time(self.baseline(y).split(yl))
        baseline[ctc_loss == 0] = 0

        # neg ctc loss as reward
        reward = -ctc_loss.detach()
        reward_bar = reward - baseline.detach()

        # reinforce loss
        rif_loss = reward_bar * nll

        nsm1_dist = self.nsm1_dist

        return {
            "ctc_loss": ctc_loss.mean(dim=0),
            "rif_loss": rif_loss.mean(dim=0),
            "ent_loss": -self.ent_coef * dist.entropy().mean(dim=0),
            "bsl_loss": F.mse_loss(baseline, reward),
        }

    def compute_loss(self, x, y):
        """
        Args:
            x: videos, [(t c h w)], i.e. list of (t c h w)
            y: labels, [(t')]
        Returns:
            losses, dict of all losses, sum then and then backward
        """
        lp = self(x)
        if self.use_sfl:
            losses = defaultdict(lambda: 0)
            for _ in range(self.monte_carlo_samples):
                for k, v in self.compute_sfl_losses(lp, y).items():
                    losses[k] += v
            for k in losses:
                losses[k] /= self.monte_carlo_samples
        else:
            s = [self.expand(yi) for yi in y]
            losses = {"ctc_loss": self.compute_ctc_loss(lp, s)}
        return losses

    def decode(self, prob, beam_width, prune, lm=None, nj=8):
        """
        Args:
            prob: [(t d)]
            beam_width: int, number of beams
            prune: minimal probability to search
            lm: probability of the last word given the prefix
            nj: number of jobs
        """
        if self.use_sfl:
            self.decoder.set_num_states(
                {i: int(n) for i, n in enumerate(self.most_probable_num_states)}
            )

        with Pool(nj) as pool:
            return list(
                pool.imap(
                    partial(
                        self.decoder.search,
                        beam_width=beam_width,
                        blank=self.blank,
                        prune=prune,
                        lm=lm,
                    ),
                    tqdm.tqdm(prob, "Decoding ..."),
                )
            )
