import tqdm
import numpy as np
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from phoenix_datasets import PhoenixVideoTextDataset, PhoenixEvaluator

import stochastic_cslr

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", default="data/phoenix-2014-multisigner")
parser.add_argument("--split", default="test")
parser.add_argument("--model", choices=["dfl", "sfl"], default="sfl")
parser.add_argument("--device", default="cuda")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--nj", type=int, default=8)
parser.add_argument("--beam-width", type=int, default=10)
parser.add_argument("--prune", type=float, default=0.01)
parser.add_argument("--use-lm", action="store_true", default=False)
args = parser.parse_args()

dataset = PhoenixVideoTextDataset(
    root=args.data_root,
    split=args.split,
    p_drop=0.5,
    random_drop=False,
    random_crop=False,
    crop_size=[224, 224],
    base_size=[256, 256],
)

data_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,  # result should be strictly ordered for evaluation.
    num_workers=args.nj,
    collate_fn=dataset.collate_fn,
)

model = stochastic_cslr.load_model(args.model == "sfl")
model.to(args.device)
model.eval()

result_dir = Path("results", args.model, args.split)
prob_path = result_dir / "prob.npz"

if args.use_lm:
    lm = dataset.corpus.create_lm()
else:
    lm = None

if prob_path.exists():
    prob = np.load(prob_path, allow_pickle=True)["prob"]
else:
    prob = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            video = list(map(lambda v: v.to(args.device), batch["video"]))
            prob += [lpi.exp().cpu().numpy() for lpi in model(video)]
    prob_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(prob_path, prob=prob)

hyp = model.decode(prob, args.beam_width, args.prune, lm, args.nj)
hyp = [" ".join([dataset.vocab[i] for i in hi]) for hi in hyp]

evaluator = PhoenixEvaluator(args.data_root)
results = evaluator.evaluate(args.split, hyp)

print(results["parsed_dtl"])

for k, v in results.items():
    if args.use_lm:
        k += "-lm"
    path = result_dir / "{k}.txt"
    with open(path, "w") as f:
        f.write(str(v))

print(f"Results has been writen to {result_dir}.")
