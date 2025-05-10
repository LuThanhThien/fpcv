from pathlib import Path
import torch
from binary_image.image import BinaryImage, config
from binary_image.modules import RasterScanner
from argparse import ArgumentParser

supported_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"]

parser = ArgumentParser(description="Test BinaryImage class")
parser.add_argument("image_name", type=str, help="Name of the image file (without extension)")
args = parser.parse_args()
img_name = Path(args.image_name)

def test_binary_image():
	img = BinaryImage.load(config.data_dir/f"{img_name}")
	img.visualize(save_path=config.output_dir/f"{img_name}-viz.png", 
				draw_centroid=True,
				draw_orientation=True,
				#   write_info_box=True,
				overwrite=True
				)

def test_raster_scanning():
    img = BinaryImage.load(config.data_dir/f"{img_name}")
    # img = BinaryImage([[1, 1, 1, 0],
    #                    [0, 0, 1, 0],
    #                    [0, 1, 1, 1],
    #                    [1, 1, 1, 1],]).to(dtype=torch.uint8)
    scanner = RasterScanner()
    labels = scanner(img)
    scanner.visualize(labels, save_path=config.output_dir/f"{img_name}-labeled.png")
    breakpoint()

def test_segmentation():
    scanner = RasterScanner()
    folder = config.data_dir / "augment"
    for img_name in folder.iterdir():
        if img_name.suffix not in supported_extensions:
            continue
        print(f"Processing {img_name}")
        img = BinaryImage.load(config.data_dir/f"{img_name}")
        labels = scanner(img)
        scanner.visualize(labels, save_path=config.output_dir/f"{img_name}-labeled.png")
    print("Done!")

if __name__=="__main__":
    # test_binary_image()
    # test_raster_scanning()
    # test_segmentation()
    pass