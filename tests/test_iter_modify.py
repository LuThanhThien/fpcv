from pathlib import Path
import torch
from torch import nn
from binary_image.image import BinaryImage, config
from binary_image.modules import InterativeModifier
from tests.test_utils import parse_args

def test_iterative_modifier(img) -> BinaryImage:
    # Initialize the InterativeModifier
    modifier = InterativeModifier()
    
    # Call the forward method
    output = modifier(img, operation=[0, 1, 0, 0])
    return output
    
args = parse_args()
img_name = Path(args.image_name) if args.image_name else ""
img = BinaryImage.load(config.data_dir/f"{img_name}")
print("Loaded image:", img_name)
out = test_iterative_modifier(img)
out.visualize(save_path=config.output_dir/f"{img_name}-viz-iter-modify.png")