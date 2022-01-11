# GCN-based-model
This repository contains pretrained models and network structure for CU partition at intra-mode.
# Dataset
We use DIV2K dataset. s. To obtain the dataset including intra CU partition modes, all images are encoded by the VVC Test Model (VTM) 7.0 at four Quantization Parameters (QPs) {22, 27, 32, 37}, and the encoder configuration is set to All-Intra mode for getting intra partition samples.
# Training
Take an example with 32x32 CU as input.
```
python main.py --train_path [PATH_TO_DATA] --num_classes 6
```
