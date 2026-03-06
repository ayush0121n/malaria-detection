"""
download_dataset.py
-------------------
Quick standalone script to download the Malaria Cell Images dataset
from Kaggle using kagglehub.

Usage:
    python download_dataset.py

Requirements:
    pip install kagglehub
    Kaggle API key at C:\\Users\\<you>\\.kaggle\\kaggle.json
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")

print("Path to dataset files:", path)
