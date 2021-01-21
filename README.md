# Stochastic CSLR

This is the PyTorch implementation for the ECCV 2020 paper: [Stochastic Fine-grained Labeling of Multi-state Sign Glosses for Continuous Sign Language Recognition](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610171.pdf).

## Quick Start

### Installation

```
pip install git+https://github.com/zheniu/stochastic-cslr
```

### Prepare the dataset

Download the RWTH-PHOENIX-2014 dataset here: [RWTH-PHOENIX-Weather 2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) and unzip it.

### Run a quick test

Run the following script for a quick test:

```python
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

model = stochastic_cslr.load_model(args.model == "sfl", pretrained=True)
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
```

By specifying the model (dfl/sfl), the data split (dev/test), whether to use lm, you will get the following result:

| Model    | WER sub/del/ins (dev) | WER sub/del/ins (test) |
| -------- | --------------------- | ---------------------- |
| dfl      | 27.1 12.7/7.4/7.0     | 27.7 13.8/7.3/6.6      |
| sfl      | 26.2 12.7/6.9/6.7     | 26.6 13.7/6.5/6.4      |
| dfl + lm | 25.6 11.5/9.2/4.9     | 26.4 12.4/9.3/4.7      |
| sfl + lm | 24.3 11.4/8.5/4.4     | 25.3 12.4/8.5/4.3      |

Note that this results are slightly different from the paper as a different random seed is used.

### Train your own model

The configuration file for deterministic and stochastic fine-grained labeling are put under `config/`. The training script is written with [torchzq](https://github.com/enhuiz/torchzq/tree/main/torchzq). The hyperparameters in the yaml file will be automatically pass to the stochastic_cslr/runner.py script. You need to either clone the repo or download the configuration files first. Before running, you need to change to the data path to the `phoenix-2014-multisigner/` folder.

#### To train (for instance, dfl):

```
tzq config/dfl-fp16.yml train
```

#### To test the trained model

```
tzq config/dfl-fp16.yml test
```

## Citation

You may cite this work by:

```
@inproceedings{niu2020stochastic,
  title={Stochastic Fine-Grained Labeling of Multi-state Sign Glosses for Continuous Sign Language Recognition},
  author={Niu, Zhe and Mak, Brian},
  booktitle={European Conference on Computer Vision},
  pages={172--186},
  year={2020},
  organization={Springer}
}
```
