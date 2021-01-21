import numpy as np
from functools import partial
from collections import defaultdict, Counter


def identity(x):
    return x


class Decoder:
    def __init__(self, vocab_size, max_num_states):
        self.vocab_size = vocab_size
        self.max_num_states = max_num_states

        # set all num states to max num states by default
        self.set_num_states(max_num_states)

    def set_num_states(self, num_states):
        if isinstance(num_states, (tuple, list, dict)):
            self.num_states = num_states
        elif type(num_states) is int:
            # tricky, no lambda should be used here as it is non-pickable
            self.num_states = defaultdict(partial(identity, num_states))
        else:
            raise TypeError(type(num_states))

    @property
    def total_states(self):
        return self.vocab_size * self.max_num_states

    def expand(self, l, num_states=None):
        """Expand a gloss sequence to state sequence"""
        l = list(map(int, l))

        if num_states is None:
            num_states = [self.num_states[g] for g in l]

        assert len(num_states) == len(l), "Length does not match."
        assert all([n > 0 for n in num_states]), "At least one state."
        assert all([n <= self.max_num_states for n in num_states]), "Max exceeded."

        return [self.g2s(g, i) for g, n in zip(l, num_states) for i in range(n)]

    def collapse(self, l):
        """Collapse a state sequence to a gloss sequence"""
        return [self.s2g(s) for s in l if self.is_ending(s)]

    def s2g(self, s):
        """State to gloss"""
        return s // self.max_num_states

    def g2s(self, g, i):
        """Gloss to state"""
        # i for order, o \in 1,2,...,max_num_states
        return g * self.max_num_states + i

    def order(self, a):
        return a % self.max_num_states

    def is_beginning(self, a):
        return self.order(a) == 0

    def is_ending(self, a):
        return self.order(a) == self.num_states[self.s2g(a)] - 1

    def is_exiting(self, a, b):
        return self.is_ending(a) and self.is_beginning(b)

    def is_next(self, a, b):
        """consecutive within one occurence of one gloss"""
        return self.s2g(a) == self.s2g(b) and self.order(b) - self.order(a) == 1

    def check_probs(self, probs):
        """
        Args:
            probs: (t tsp1)
        """
        assert (
            probs.shape[1] == self.total_states + 1
        ), f"probs.shape[1] should be total_states + 1, but got shape: {probs.shape}."

        assert 0 <= probs.min() and probs.max() <= 1, "probs should be within [0, 1]."

    def successors(self, s):
        """To avoid the case where all possible paths are pruned."""
        ret = {s}
        if not self.is_ending(s):
            ret.add(self.g2s(self.s2g(s), self.order(s) + 1))
        return ret

    def search(self, probs, beam_width, blank, prune, lm, alpha=0.3):
        self.check_probs(probs)

        if lm is None:
            lm = lambda *_: 1

        def mslm(l):
            if len(l) == 1:
                return self.is_beginning(l[-1])
            a, b = l[-2:]
            if self.is_next(a, b):
                return 1
            elif self.is_exiting(a, b):
                return lm(self.collapse(l)) ** alpha
            return 0

        p_b = defaultdict(Counter)
        p_nb = defaultdict(Counter)

        p_b[-1][()] = 1
        p_nb[-1][()] = 0

        prefixes = [()]

        for t in range(len(probs)):
            pruned_states, prune_relaxed = [], prune
            while not pruned_states:
                pruned_states = np.where(probs[t] >= prune_relaxed)[0].tolist()
                prune_relaxed /= 2
            pruned_states = set(pruned_states)

            for l in prefixes:
                possible_states = {blank} | pruned_states
                if l:
                    possible_states |= self.successors(l[-1])
                for s in possible_states:
                    p_t_s = probs[t, s]

                    if s == blank:
                        p_b[t][l] += p_t_s * (p_b[t - 1][l] + p_nb[t - 1][l])
                        continue

                    ls = l + (s,)
                    p_lm = mslm(ls)

                    if l and s == l[-1]:
                        # a_ + a = aa
                        p_nb[t][ls] += p_lm * p_t_s * p_b[t - 1][l]

                        # a + a = a
                        p_nb[t][l] += p_t_s * p_nb[t - 1][l]
                    else:
                        # a(_) + b = ab
                        p_nb[t][ls] += p_lm * p_t_s * (p_b[t - 1][l] + p_nb[t - 1][l])

                    # if ls not in prefixes:
                    #     p_b[t][ls] += probs[t, blank] * (
                    #         p_b[t - 1][ls] + p_nb[t - 1][ls]
                    #     )

                    #     p_nb[t][ls] += p_t_s * p_nb[t - 1][ls]

            p = p_b[t] + p_nb[t]

            if len(p) == 0:
                p = p_b[t]  # 0 prob for all prefix

            if len(p) == 0:
                p = p_nb[t]  # 0 prob for all prefix

            prefixes = sorted(p, key=lambda k: p[k], reverse=True)
            prefixes = prefixes[:beam_width]

            # divide by a constant (min_prob) to avoid underflow
            min_prob = np.inf
            for prefix in prefixes:
                if min_prob > p[prefix] and p[prefix] > 0:
                    min_prob = p[prefix]
            for prefix in prefixes:
                # usually, min_prob won't be zero
                p_b[t][prefix] /= min_prob
                p_nb[t][prefix] /= min_prob

            if p[prefixes[0]] == 0:
                raise ValueError("Even the most probable beam has probability 0. ")

        hyp = self.collapse(prefixes[0])

        return hyp
