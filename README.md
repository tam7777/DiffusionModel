# Generating Dermatological Images using Diffusion Transformer

This repository contains the code and data for the project on generating synthetic dermatological images using Diffusion Transformer (DiT). The goal is to enhance medical image datasets and improve AI model training in dermatology. This work builds on Diffusion Transformers to create high-quality synthetic images for both standardized datasets and in-the-wild dermatological conditions.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Model Details](#model-details)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this project, we leverage **Diffusion Transformers (DiT)** to generate synthetic dermatological images. Traditional models are often hindered by biased datasets and limited image diversity. DiT helps address these challenges, especially by generating images for underrepresented categories in dermatological datasets.

## Installation

To run this project, using venv is recommended. Once you have activated venv, install the following dependencies:

```bash
pip install -r requirements.txt
```

## Usage
For simplicity, training and evaluation is done together. All you need to do is run the python code below. Depending on the number of machine you are using, change the number of --nproc_per_node. The code is modified so that it works efficiently on H100.
```
torchrun --standalone --nproc_per_node=1 main.py
```
The details of the model is specified at config.yaml, including which dataset to use or which model to use


## Datasets

## Model Details

## Performance

## Contributing

## License

