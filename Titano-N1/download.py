import os
from PIL import Image
from io import BytesIO
import time
import requests
import json
import urllib.parse
import random
import re
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

def setup_folders():
    base_dir = 'dataset'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    return train_dir, val_dir

def get_search_urls(keywords):
    search_term = urllib.parse.quote(keywords)
    return [
        f"https://www.google.com/search?q={search_term}&tbm=isch",
        f"https://www.bing.com/images/search?q={search_term}",
        f"https://images.search.yahoo.com/search/images?p={search_term}"
    ]

def extract_image_urls(html_content):
    image_urls = set()
    
    patterns = [
        r'https?://[^"\']+\.(?:jpg|jpeg|png|gif)',
        r'https?://[^"\']+\.(?:JPG|JPEG|PNG|GIF)',
        r'"url":"(https?://[^"]+\.(?:jpg|jpeg|png|gif))"',
        r'&imgurl=(https?://[^&]+\.(?:jpg|jpeg|png|gif))'
    ]
    
    soup = BeautifulSoup(html_content, 'html.parser')
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if src.startswith('http') and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
            image_urls.add(src)
            
    for pattern in patterns:
        urls = re.findall(pattern, html_content, re.IGNORECASE)
        image_urls.update(urls if isinstance(urls, (list, set)) else [urls])
    
    return list(image_urls)

def search_images(keywords, max_results=100):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    image_urls = set()
    search_urls = get_search_urls(keywords)
    
    for url in search_urls:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                urls = extract_image_urls(response.text)
                image_urls.update(urls)
        except Exception as e:
            print(f"Error searching {url}: {e}")
        time.sleep(2)
    
    return list(image_urls)[:max_results]

def verify_image(img_url, headers):
    try:
        response = requests.head(img_url, headers=headers, timeout=5)
        content_type = response.headers.get('content-type', '')
        content_length = response.headers.get('content-length', '0')
        
        return (
            response.status_code == 200 and
            'image' in content_type and
            int(content_length) > 10000
        )
    except:
        return False

def download_images(search_term, folder, num_images):
    os.makedirs(folder, exist_ok=True)
    count = 0
    
    image_urls = search_images(search_term, max_results=num_images*3)
    
    if not image_urls:
        print(f"No images found for {search_term}")
        return count
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.google.com'
    }
    
    valid_urls = [url for url in image_urls if verify_image(url, headers)]
    
    for idx, img_url in enumerate(valid_urls):
        if count >= num_images:
            break
            
        try:
            image_data = requests.get(img_url, timeout=10, headers=headers)
            img_obj = Image.open(BytesIO(image_data.content))
            
            if img_obj.size[0] < 200 or img_obj.size[1] < 200:
                continue
                
            if img_obj.mode != 'RGB':
                img_obj = img_obj.convert('RGB')
                
            img_path = os.path.join(folder, f"{search_term}_{idx}.jpg")
            img_obj.save(img_path, quality=85, optimize=True)
            count += 1
            print(f"Downloaded {count}/{num_images} images for {search_term}")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error downloading image: {e}")
            continue
            
    return count

def main():
    print("Image Downloader for AI Training")
    print("--------------------------------")
    
    train_dir, val_dir = setup_folders()
    
    print("\nEnter categories (one per line, press enter twice when done):")
    categories = []
    while True:
        category = input()
        if category == "":
            break
        categories.append(category)
    
    if not categories:
        print("No categories provided. Exiting...")
        return
    
    try:
        num_images = int(input("\nHow many images per category? "))
        val_split = float(input("Validation split (0.0-1.0)? "))
    except ValueError:
        print("Invalid input. Using defaults: 50 images, 0.2 validation split")
        num_images = 50
        val_split = 0.2
    
    for category in categories:
        print(f"\nDownloading images for category: {category}")
        
        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        
        train_count = int(num_images * (1 - val_split))
        val_count = num_images - train_count
        
        print("Downloading training images...")
        download_images(category, train_category_dir, train_count)
        
        print("Downloading validation images...")
        download_images(category, val_category_dir, val_count)

if __name__ == "__main__":
    main()
