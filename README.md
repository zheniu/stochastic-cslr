# Stochastic CSLR

This is the PyTorch implementation for the ECCV 2020 paper: [Stochastic Fine-grained Labeling of Multi-state Sign Glosses for Continuous Sign Language Recognition](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610171.pdf).

## Quick Start

### 1. Installation

```
pip install git+https://github.com/zheniu/stochastic-cslr
```

Also, please install `sclite` and put it in your PATH for evaluation. Please take a look at step 2 for instructions.

### 2. Prepare the dataset

Download the RWTH-PHOENIX-2014 dataset [here](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/), unzip it and obtain the path to `phoenix-2014-multisigner/` folder for later use.

Under the dataset folder, you can find a file `phoenix-2014-multisigner/evaluation/NIST-sclite_sctk-2.4.0-20091110-0958.tar.bz2`, this helps you install `sclite`, after installation, put `sclite` into your PATH for later evaluation.

### 3. Run a quick test

You may download the script `quick_test.py` and put it directory you want for a quick test.

```
python3 quick_test.py --data-root your_path_to/phoenix-2014-multisigner
```

By specifying the model type `--model sfl/dfl`, the data split `--split dev/test`, whether to use lm `--use-lm`, you can get the following results:

| Model    | WER sub/del/ins (dev) | WER sub/del/ins (test) |
| -------- | --------------------- | ---------------------- |
| dfl      | 27.1 12.7/7.4/7.0     | 27.7 13.8/7.3/6.6      |
| sfl      | 26.2 12.7/6.9/6.7     | 26.6 13.7/6.5/6.4      |
| dfl + lm | 25.6 11.5/9.2/4.9     | 26.4 12.4/9.3/4.7      |
| sfl + lm | 24.3 11.4/8.5/4.4     | 25.3 12.4/8.5/4.3      |

Note that these results are slightly different from the paper as a different random seed is used.

### 4. Train your own model

The configuration files for deterministic and stochastic fine-grained labeling are put under `config/`. The training script is based on a PyTorch experiment runner [torchzq](https://github.com/enhuiz/torchzq/tree/main/torchzq). The hyperparameters in the yaml file will be automatically pass to the `stochastic_cslr/runner.py` script. You need to either clone the repo or download the configuration files first. Before running, you need to change to the data path to the `phoenix-2014-multisigner/` folder.

#### Train (for instance, dfl):

```
tzq config/dfl-fp16.yml train
```

#### Test the trained model

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
