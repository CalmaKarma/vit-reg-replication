# vit-reg-replication

This repo is an attempt to replicate the paper [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588).

# Replication Goals

1. Use pretrained checkpoints of ViTs w/o and w/ registers to show L2 norm distribution and attention map 
of the most extreme outliers. 
2. Try to finetune ViTs with registers instead of training from scratch to replicate the performance enhancement by registers.

# Prepare ImageNet Dataset

We only used ImageNet's validation dataset as our whole dataset due to limit of time and computational resource.

1. Download and decompress ImageNet's ILSVRC2012's validation dataset to `ILSVRC2012/ILSVRC2012_img_val`.
2. Download and decompress ImageNet's ILSVRC2012's devkit to `ILSVRC2012/ILSVRC2012_devkit_t12`.
3. Run `imagenet_preprocess.py` to categorize the dataset based on classification.
4. Run `iamgenet_val_split.py` to split the dataset into 80% training and 20% validation.

# Finetuning OpenCLIP with Registers

Run `openclip_with_registers.py` to finetune OpenCLIP with registers on ImageNet.

Resulting checkpoints with the outputted to `outputs`.

For hyperparameters, please check the arguments required in the code.

# Generate L2 Norm Distribution Charts and Attention Maps

Run `analyze_mdels.ipynb` for OpenCLIP from OpenCLIP.

Run `analyze_models_dinov2` for DINOv2 from torch hub.