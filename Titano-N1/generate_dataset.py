import os
from captcha.image import ImageCaptcha
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_captcha_dataset(train_dir, val_dir, num_images=1000):
    """Generate synthetic captcha images"""
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    image = ImageCaptcha(width=200, height=80)
    chars = string.ascii_letters + string.digits
    
    # Split images between train and validation
    train_size = int(num_images * 0.8)
    
    print(f"Generating {num_images} captcha images...")
    
    for i in range(num_images):
        # Generate random text
        text = ''.join(random.choices(chars, k=random.randint(4, 8)))
        
        # Determine output directory
        out_dir = train_dir if i < train_size else val_dir
        
        # Generate and save captcha
        img_path = os.path.join(out_dir, f'captcha_{i}_{text}.png')
        image.write(text, img_path)
        
        if i % 100 == 0:
            print(f"Generated {i}/{num_images} images")
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    train_dir = 'dataset/train/captcha text'
    val_dir = 'dataset/val/captcha text'
    generate_captcha_dataset(train_dir, val_dir)
