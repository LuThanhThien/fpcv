from pathlib import Path
import torch
from binary_image.image import BinaryImage
from argparse import ArgumentParser

supported_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"]
sample_img = BinaryImage([[[0, 1, 1, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],]]).to(dtype=torch.uint8)

def parse_args():
    parser = ArgumentParser(description="Test BinaryImage class")
    parser.add_argument("--image_name", "-i", type=str, default=None, help="Name of the image file (without extension)")
    args = parser.parse_args()
    return args
