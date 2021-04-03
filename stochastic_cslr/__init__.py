import os
import gdown
import torch
from pathlib import Path

from .model import Model

gdrive_file_ids = {
    "dfl": "1jgf4pmsWoTeXTman7Sd0Yi-anhpUiQ_Y",
    "sfl": "1dkE9MNflNQYJF4jdLpnM2rjuoBroaZdL",
}


def download(name):
    url = f"https://drive.google.com/uc?id={gdrive_file_ids[name]}"
    path = Path(torch.hub.get_dir(), "stochastic_cslr", name).with_suffix(".pth")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(path), quiet=False)
    return path


def load_model(use_sfl=True, pretrained=True):
    model = Model(
        vocab_size=1232,
        dim=512,
        max_num_states=5 if use_sfl else 2,
        use_sfl=use_sfl,
    )

    while True:
        path = download("sfl" if use_sfl else "dfl")
        try:
            model.load_state_dict(torch.load(path, "cpu"))
            break
        except Exception as e:
            print(e)
            print("Fail to load model, re-downloading ...")
            os.remove(path)
    return model
