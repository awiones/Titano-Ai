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

def construct_search_query(category, is_text=False):
    """Create more specific search queries for text and handwriting"""
    if is_text:
        # Base terms for different text categories
        text_base_terms = {
            'handwriting': [
                "handwritten text sample",
                "handwriting example",
                "written notes sample",
                "handwritten letter",
                "cursive writing sample",
                "handwritten words",
            ],
            'captcha': [
                "captcha verification example",
                "website captcha",
                "text captcha example",
                "captcha test image",
                "security captcha",
            ],
            'text': [
                "printed text sample",
                "text document image",
                "text paragraph sample",
                "text snippet image",
                "text block example",
            ]
        }
        
        # Determine which category we're dealing with
        category_lower = category.lower()
        if 'handwrit' in category_lower or 'cursive' in category_lower:
            base_terms = text_base_terms['handwriting']
        elif 'captcha' in category_lower:
            base_terms = text_base_terms['captcha']
        else:
            base_terms = text_base_terms['text']
            
        # Add category-specific variations
        variations = base_terms + [
            f"{category} example",
            f"{category} sample",
            f"{category} image",
            category
        ]
    else:
        variations = [
            f"{category} high quality photo",
            f"{category} professional picture",
            f"{category} clear image",
            f"{category} high resolution",
            f"{category} detailed photo",
            category
        ]
    
    return variations

def get_search_urls(keywords):
    """Enhanced search URLs with text-specific parameters"""
    search_term = urllib.parse.quote(keywords)
    return [
        f"https://www.google.com/search?q={search_term}&tbm=isch&tbs=ic:specific,itp:text",
        f"https://www.google.com/search?q={search_term}&tbm=isch&tbs=ic:gray,itp:text",
        f"https://www.bing.com/images/search?q={search_term}&qft=+filterui:photo-text",
        f"https://www.google.com/search?q={search_term}&tbm=isch"
    ]

def is_relevant_image(url, category_terms):
    """Check if image URL contains relevant keywords"""
    url_lower = url.lower()
    return any(term.lower() in url_lower for term in category_terms)

def extract_image_urls(html_content, category_terms):
    """Enhanced image URL extraction for text images"""
    image_urls = set()
    
    # Expanded patterns for text images
    patterns = [
        r'https?://[^"\']+\.(?:jpg|jpeg|png|gif|bmp)',
        r'https?://[^"\']+\.(?:JPG|JPEG|PNG|GIF|BMP)',
        r'"url":"(https?://[^"]+\.(?:jpg|jpeg|png|gif|bmp))"',
        r'&imgurl=(https?://[^&]+\.(?:jpg|jpeg|png|gif|bmp))',
        r'data-src="(https?://[^"]+\.(?:jpg|jpeg|png|gif|bmp))"',
        r'content="(https?://[^"]+\.(?:jpg|jpeg|png|gif|bmp))"'
    ]
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find images with text-related attributes
    for img in soup.find_all(['img', 'source']):
        for attr in ['src', 'data-src', 'srcset', 'data-srcset', 'data-original']:
            src = img.get(attr, '')
            if src.startswith('http'):
                # Check for text-related indicators in alt text or surrounding elements
                alt_text = img.get('alt', '').lower()
                parent_text = img.parent.get_text().lower() if img.parent else ''
                
                if (any(term.lower() in src.lower() for term in category_terms) or
                    any(term.lower() in alt_text for term in category_terms) or
                    any(term.lower() in parent_text for term in category_terms)):
                    image_urls.add(src)
    
    # Apply regex patterns
    for pattern in patterns:
        urls = re.findall(pattern, html_content, re.IGNORECASE)
        for url in (urls if isinstance(urls, (list, set)) else [urls]):
            if url and is_relevant_image(url, category_terms):
                image_urls.add(url)
    
    return list(image_urls)

def search_images(category, max_results=100):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    image_urls = set()
    category_terms = category.split()
    
    # Try different search queries
    for query in construct_search_query(category):
        search_urls = get_search_urls(query)
        
        for url in search_urls:
            retries = 3
            while retries > 0:
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                    if response.status_code == 200:
                        urls = extract_image_urls(response.text, category_terms)
                        image_urls.update(urls)
                        break
                except Exception as e:
                    print(f"Error searching {url}: {e}")
                    retries -= 1
                    time.sleep(1)
            time.sleep(2)
    
    return list(set(image_urls))[:max_results]

def verify_image(img_url, headers, min_size=(200, 200), max_size=(4000, 4000)):
    """Simplified image verification"""
    try:
        # Get the image directly
        img_response = requests.get(img_url, headers=headers, timeout=5)
        if img_response.status_code != 200:
            return False

        img = Image.open(BytesIO(img_response.content))
        
        # Basic size check
        width, height = img.size
        if not (min_size[0] <= width <= max_size[0] and 
               min_size[1] <= height <= max_size[1]):
            return False
            
        # Ensure it's an image we can work with
        if img.format not in ['JPEG', 'PNG', 'BMP']:
            return False

        return True

    except:
        return False

def download_images(search_term, folder, num_images, is_text=False):
    """Enhanced download function that continues until target count is reached"""
    os.makedirs(folder, exist_ok=True)
    count = 0
    page = 0
    max_retries = 10  # Maximum number of search page retries
    
    print(f"\nAttempting to download {num_images} images")
    print(f"Target folder: {folder}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }
    
    # Set size requirements
    if is_text:
        min_size = (50, 20)
        max_size = (800, 400)
    else:
        min_size = (200, 200)
        max_size = (4000, 4000)
    
    while count < num_images and page < max_retries:
        # Get more image URLs
        print(f"\rSearching page {page + 1} for more images...", end="")
        image_urls = search_images(f"{search_term} page:{page}", max_results=num_images*2)
        
        if not image_urls:
            page += 1
            continue
        
        print(f"\nFound {len(image_urls)} potential URLs")
        
        for img_url in image_urls:
            if count >= num_images:
                break
                
            try:
                print(f"\rDownloading: {count}/{num_images}", end="")
                
                # Skip verification for known good domains
                should_verify = not any(domain in img_url.lower() for domain in [
                    'googleusercontent.com',
                    'ggpht.com',
                    'imgur.com',
                    'wikimedia.org'
                ])
                
                if should_verify and not verify_image(img_url, headers, min_size, max_size):
                    continue
                    
                image_data = requests.get(img_url, timeout=5, headers=headers)
                img = Image.open(BytesIO(image_data.content))
                
                # Convert to grayscale for text images
                if is_text:
                    img = img.convert('L')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save image
                img_path = os.path.join(folder, f"{search_term}_{count}.jpg")
                img.save(img_path, 'JPEG', quality=85, optimize=True)
                count += 1
                
                if count >= num_images:
                    break
                    
            except Exception as e:
                continue
        
        page += 1
        
        # If we still need more images, wait a bit before next search
        if count < num_images:
            print(f"\nStill need {num_images - count} more images. Continuing search...")
            time.sleep(2)
    
    print(f"\nDownloaded {count}/{num_images} images")
    
    # If we couldn't get enough images, ask user what to do
    if count < num_images:
        print("\nCouldn't find enough images. Options:")
        print("1. Continue searching with different terms")
        print("2. Accept current number of images")
        print("3. Generate synthetic images (for text/captcha)")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            # Try with alternative search terms
            remaining = num_images - count
            alt_terms = [f"different {search_term}", f"alternative {search_term}", f"more {search_term}"]
            
            for term in alt_terms:
                if count >= num_images:
                    break
                count += download_images(term, folder, remaining, is_text)
                remaining = num_images - count
                
        elif choice == '3' and is_text and 'captcha' in search_term.lower():
            # Generate synthetic captchas
            print("\nGenerating synthetic captcha images...")
            from generate_captchas import generate_captcha_dataset
            generate_captcha_dataset(num_images - count)
            count = num_images
    
    return count

def main():
    print("Image Downloader for AI Training")
    print("--------------------------------")
    
    train_dir, val_dir = setup_folders()
    
    print("\nSelect download type:")
    print("1. Regular Images (Animals, Objects)")
    print("2. Text Images (Words, Characters)")
    
    while True:
        mode = input("\nEnter your choice (1-2): ").strip()
        if mode in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    is_text = mode == '2'
    
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
        print(f"\nProcessing category: {category}")
        print("=" * 50)
        
        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        
        train_count = int(num_images * (1 - val_split))
        val_count = num_images - train_count
        
        print(f"\nTraining set ({train_count} images):")
        downloaded_train = download_images(category, train_category_dir, train_count, is_text=is_text)
        
        print(f"\nValidation set ({val_count} images):")
        downloaded_val = download_images(category, val_category_dir, val_count, is_text=is_text)
        
        print(f"\nSummary for {category}:")
        print(f"Training images: {downloaded_train}/{train_count}")
        print(f"Validation images: {downloaded_val}/{val_count}")
        print("=" * 50)

if __name__ == "__main__":
    main()
