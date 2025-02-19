from captcha.image import ImageCaptcha
import os
import random
import string

def generate_captcha_dataset(num_images=1000):
    output_dir = 'dataset/train/captcha text'
    val_dir = 'dataset/val/captcha text'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    image = ImageCaptcha(width=200, height=80)
    chars = string.ascii_letters + string.digits

    for i in range(num_images):
        # Generate random text
        text = ''.join(random.choices(chars, k=random.randint(4, 8)))
        
        # Generate captcha image
        img_path = os.path.join(
            val_dir if i < num_images * 0.2 else output_dir,
            f'captcha_{i}_{text}.png'
        )
        image.write(text, img_path)
        
        if i % 100 == 0:
            print(f"Generated {i} captchas...")

if __name__ == "__main__":
    generate_captcha_dataset()
