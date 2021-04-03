from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="stochastic_cslr",
    python_requires=">=3.6.0",
    version="0.0.1.dev0",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["stochastic_cslr"],
    install_requires=[
        "phoenix_datasets @  git+https://github.com/enhuiz/phoenix-datasets",
        "tqdm",
        "pandas",
        "numpy",
        "xmltodict",
        "einops",
        "torch==1.7.0",
        "torchvision==0.8.1",
        "torchzq==1.0.6",
        "tensorboard",
        "imageio",
        "gdown",
    ],
    url="https://github.com/enhuiz/stochastic_cslr",
)
