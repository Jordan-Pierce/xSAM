# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="x_segment_anything",
    version="0.0.6",
    author="Jordan Pierce",
    author_email="jordan.pierce@noaa.gov",
    url="https://github.com/Jordan-Pierce/xSAM",
    python_requires=">=3.8",
    install_requires=["torch", "torchvision"],
    packages=find_packages(exclude="checkpoints"),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
)
