from binary_image.image import generate_multi_object_images, config
from argparse import ArgumentParser

if __name__=="__main__":
    parser = ArgumentParser(description="Generate augmented images")
    parser.add_argument("--num_images", type=int, default=100, help="Number of generated images")
    parser.add_argument("--max_objects", type=int, default=5, help="Maximum number of objects per frame")
        
    args = parser.parse_args()
    
    generate_multi_object_images(
        config.data_dir,
        num_images=args.num_images,
        max_objects=args.max_objects,
    )