#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fl_pytorch",
    version="0.0.1",
    author_email="konstantin.burlachenko@kaust.edu.sa,samuel.horvath@kaust.edu.sa,peter.richtarik@kaust.edu.sa",
    description="Efficient Simulated Federated Learning Experimental"
                "Enviroment based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/burlachenkok/flpytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
