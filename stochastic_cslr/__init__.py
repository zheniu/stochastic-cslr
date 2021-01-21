from torch.utils.model_zoo import load_url

from .model import Model

model_urls = {
    "dfl": "https://github.com/zheniu/stochastic-cslr-ckpt/raw/main/dfl.pth",
    "sfl": "https://github.com/zheniu/stochastic-cslr-ckpt/raw/main/sfl.pth",
}


def load_model(use_sfl=True, pretrained=True):
    model = Model(
        vocab_size=1232,
        dim=512,
        max_num_states=5 if use_sfl else 2,
        use_sfl=use_sfl,
    )
    if pretrained:
        model.load_state_dict(load_url(model_urls["sfl" if use_sfl else "dfl"]))
    return model
