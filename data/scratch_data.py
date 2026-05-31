
import requests
from bs4 import BeautifulSoup
import re
import os
import time
from urllib.parse import quote

def download_bing_images(search_term, count=10, save_dir='bing_images'):

  
    # create/save directories
    os.makedirs(save_dir, exist_ok=True)
    
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    # encode search term
    encoded_term = quote(search_term)
    
    # build URL
    base_url = "https://cn.bing.com/images/search" #where to search
    
    downloaded = 0
    page_size = 35  
    pages_needed = (count + page_size - 1) // page_size
    
    for page in range(pages_needed):
        if downloaded >= count:
            break
            
       
        offset = page * page_size
        url = f"{base_url}?q={encoded_term}&first={offset}"
        
        try:
            print(f"is getting page {page+1} ...")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            
            image_patterns = [
                {'tag': 'a', 'class_': 'iusc'},
                {'tag': 'div', 'class_': 'imgpt'},
                {'tag': 'li', 'class_': 'imgpt'},
            ]
            
            image_elements = []
            for pattern in image_patterns:
                elements = soup.find_all(pattern['tag'], class_=pattern.get('class_'))
                if elements:
                    image_elements = elements
                    break
            
            if not image_elements:
                
                img_tags = soup.find_all('img', {'src': re.compile(r'https?://')})
                for img in img_tags[:count-downloaded]:
                    try:
                        img_url = img.get('src')
                        if img_url and 'http' in img_url:
                            download_image(img_url, downloaded, search_term, save_dir)
                            downloaded += 1
                            time.sleep(0.5)  
                    except Exception as e:
                        print(f"download failed: {e}")
                        continue
                continue
            
            
            for element in image_elements:
                if downloaded >= count:
                    break
                    
                try:
                   
                    m_attr = element.get('m')
                    if m_attr:
                        import json
                        img_info = json.loads(m_attr)
                        img_url = img_info.get('murl')
                        
                        if img_url:
                            download_image(img_url, downloaded, search_term, save_dir)
                            downloaded += 1
                            time.sleep(0.5)  
                            
                except Exception as e:
                    print(f"Parse failed: {e}")
                    continue
                    
        except Exception as e:
            print(f"Failed to retrieve the page: {e}")
            continue

def download_image(img_url, index, search_term, save_dir):
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(img_url, headers=headers, timeout=10, stream=True)
        response.raise_for_status()
        
        
        content_type = response.headers.get('content-type', '')
        if 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        elif 'png' in content_type:
            ext = '.png'
        elif 'gif' in content_type:
            ext = '.gif'
        elif 'webp' in content_type:
            ext = '.webp'
        else:
            ext = '.jpg'
        
        # save files
        file_name = f"{search_term}_{index+1}{ext}"
        file_path = os.path.join(save_dir, file_name)
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        
        print(f"downloaded successfully: {file_name}")
        
    except Exception as e:
        print(f"Images Download fails {img_url}: {e}")

# Example
if __name__ == "__main__":
    
    tag = "Kung_Fu_Panda"
    download_bing_images(tag, count=20, save_dir=f'/Users/stevezhou/Desktop/LoRA/{tag}') 
