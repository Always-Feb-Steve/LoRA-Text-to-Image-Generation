#爬虫
import requests
from bs4 import BeautifulSoup
import re
import os
import time
from urllib.parse import quote

def download_bing_images(search_term, count=10, save_dir='bing_images'):
    """
    从Bing图片搜索下载图片
    注意：请遵守Bing的robots.txt和使用条款
    """
    # create/save directories
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置请求头，模拟浏览器
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
    page_size = 35  # Bing每页显示的大概图片数
    pages_needed = (count + page_size - 1) // page_size
    
    for page in range(pages_needed):
        if downloaded >= count:
            break
            
        # 计算偏移量
        offset = page * page_size
        url = f"{base_url}?q={encoded_term}&first={offset}"
        
        try:
            print(f"正在获取第 {page+1} 页...")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找图片链接 - 注意：Bing的HTML结构可能会变化
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
                # 尝试直接搜索img标签
                img_tags = soup.find_all('img', {'src': re.compile(r'https?://')})
                for img in img_tags[:count-downloaded]:
                    try:
                        img_url = img.get('src')
                        if img_url and 'http' in img_url:
                            download_image(img_url, downloaded, search_term, save_dir)
                            downloaded += 1
                            time.sleep(0.5)  # 礼貌延迟
                    except Exception as e:
                        print(f"download failed: {e}")
                        continue
                continue
            
            # 从m属性中提取图片信息
            for element in image_elements:
                if downloaded >= count:
                    break
                    
                try:
                    # Bing将图片信息存储在m属性中
                    m_attr = element.get('m')
                    if m_attr:
                        import json
                        img_info = json.loads(m_attr)
                        img_url = img_info.get('murl')
                        
                        if img_url:
                            download_image(img_url, downloaded, search_term, save_dir)
                            downloaded += 1
                            time.sleep(0.5)  # 礼貌延迟
                            
                except Exception as e:
                    print(f"解析失败: {e}")
                    continue
                    
        except Exception as e:
            print(f"获取页面失败: {e}")
            continue

def download_image(img_url, index, search_term, save_dir):
    """下载单个图片"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(img_url, headers=headers, timeout=10, stream=True)
        response.raise_for_status()
        
        # 根据Content-Type确定文件扩展名
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
        
        # 保存文件
        file_name = f"{search_term}_{index+1}{ext}"
        file_path = os.path.join(save_dir, file_name)
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        
        print(f"已下载: {file_name}")
        
    except Exception as e:
        print(f"下载图片失败 {img_url}: {e}")

# 使用示例
if __name__ == "__main__":
    # 注意：使用前请确保遵守Bing的使用条款
    tag = "Kung_Fu_Panda"
    download_bing_images(tag, count=20, save_dir=f'/Users/stevezhou/Desktop/LoRA/{tag}') #下20张图片，save_direc 到