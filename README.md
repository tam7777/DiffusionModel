# Generating Dermatological Images using Diffusion Transformer

This repository contains the code for the project on generating synthetic dermatological images using Diffusion Transformer (DiT). The goal is to enhance medical image datasets and improve AI model training in dermatology. This work builds on Diffusion Transformers to create high-quality synthetic images for both standardized datasets and in-the-wild dermatological conditions. 

Modifications are needed as the current explanation does not fully reflect or explain the methodologies and findings presented in the accompanying paper. 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Model Details](#model-details)
- [Performance](#performance)
- [License](#license)

## Introduction

In this project, we leverage **Diffusion Transformers (DiT)** to generate synthetic dermatological images. Traditional models are often hindered by biased datasets and limited image diversity. DiT helps address these challenges, especially by generating images for underrepresented categories in dermatological datasets.

## Installation

To run this project, using venv is recommended. Once you have activated venv, install the following dependencies:

```bash
pip install -r requirements.txt
```

## Usage
For simplicity, training and evaluation is done together. All you need to do is run the python code below. Depending on the number of machine you are using, change the number of `--nproc_per_node`. The code is modified so that it works efficiently on H100.
```
torchrun --standalone --nproc_per_node=1 main.py
```
The details of the model is specified at config.yaml, including which dataset to use or which model to use


## Datasets

| Dataset | Description | Source |
| --- | --- | --- |
| HAM10000 | Large collection of dermatoscopic skin lesion images | [HAM10000](https://www.nature.com/articles/sdata2018161) |
| SCIN | In-the-wild images of various skin conditions | [SCIN](https://research.google/blog/scin-a-new-resource-for-representative-dermatology-images/) |

## Model Details
Three models were trained and compared in this study:
| Model | Description |
| --- | --- |
| DiT | Normal DiT model for image generation |
| DiT-B | Enhanced DiT with more attention heads |
| LDM | Latent Diffusion Model baseline |

## Performance
The table below shows the performance of different models measured using Frechet Inception Distance (FID) after training:
HAM10000:
| Model | FID (900 epochs) |
| --- | --- |
| DiT | 206.246 |
| DiT-B | 163.236 |
| LDM | 220.160 |

SCIN Dataset:
| Model | FID (2500 epochs) |
| --- | --- |
| DiT | 322.518 |
| DiT-B | 293.374 |


**Results**
The DiT-B model demonstrated superior performance on the HAM10000 dataset, producing high-quality, realistic dermatological images. However, further optimization is required to improve performance on the more complex SCIN dataset.

## License
This project is licensed under the MIT License.
