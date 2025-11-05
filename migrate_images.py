import os, re, shutil, requests, hashlib
import subprocess, sys, argparse
from urllib.parse import urlparse


DOCS_DIR = 'D:\\Project\\NewBlog\\docs'
IMAGE_REGIX = r'!\[(.*?)\]\((.*?)\)'

def get_unique_filename(target_dir, filename):
    """
    防止命名重复, 如果重复，会在文件名后加(x)
    返回不重复的文件名
    """
    if not os.path.exists(os.path.join(target_dir, filename)):
        return filename
    base, ext = os.path.splitext(filename)
    cnt = 1
    new_filename = f"{base}({cnt}){ext}"
    while os.path.exists(os.path.join(target_dir, new_filename)):
        cnt += 1
        new_filename = f"{base}({cnt}){ext}"
    return new_filename



def download_image(url, target_dir):
    """
    从url下载图片到target_dir
    返回新的文件名，如果下载失败返回None
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        path = urlparse(url).path
        filename = os.path.basename(path)
        if not filename or '.' not in filename:
            content_type = response.headers.get('content-type')
            exts = ['jpg', 'jpeg', 'png', 'gif', 'webp']
            ext = '.jpg'
            for x in exts:
                if content_type and x in content_type:
                    ext = '.' + x
            filename = hashlib.md5(url.encode('utf-8')).hexdigest() + ext
        unique_filename = get_unique_filename(target_dir, filename)
        target_path = os.path.join(target_dir, unique_filename)
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        print(f"[下载] {url} -> {target_path}")
        return unique_filename
    except requests.exceptions.RequestException as e:
        print(f"[下载失败] {url} (Error: {e})")
        return None

def copy_image(src_path, target_dir):
    """
    从本地绝对路径复制文件到target_dir
    返回新文件名，失败返回None
    """
    if not os.path.exists(src_path):
        print(f"[复制失败] 绝对路径不存在: {src_path}")
        return None
    try:
        filename = os.path.basename(src_path)
        unique_filename = get_unique_filename(target_dir, filename)
        target_path = os.path.join(target_dir, unique_filename) 
        shutil.copy2(src_path, target_path)
        print(f"[复制] {src_path} -> {target_path}")
        return unique_filename
    except Exception as e:
        print(f"[复制失败] {src_path} (Error: {e})")
        return None

def process_markdown_file(md_path):
    print(f"\n--- 正在处理: {md_path} ---")
    md_dir = os.path.dirname(md_path)
    images_dir = os.path.join(md_dir, "images")
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read() 
    except Exception as e:
        print(f"[错误] 无法读取文件: {e}")
        return
    
    old_content = content
    
    def replacer(match):
        alt_text = match.group(1)
        old_path = match.group(2)
        new_filename = None
        os.makedirs(images_dir, exist_ok=True)
        if old_path.startswith("http://") or old_path.startswith("https://"):
            new_filename = download_image(old_path, images_dir)
        elif os.path.isabs(old_path):
            new_filename = copy_image(old_path, images_dir)
        else:
            print(f"[跳过] 相对路径: {old_path}")
            return match.group(0)
        if new_filename:
            new_md_path = f"images/{new_filename}"
            return f"![{alt_text}]({new_md_path})"
        else:
            return match.group(0)
    
    new_content = re.sub(IMAGE_REGIX, replacer, content)

    if new_content != old_content:
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(new_content) 
            print(f"[完成] 已更新文件: {md_path}")
        except Exception as e:
            print(f"[错误] 无法写回文件: {e}")
    else:
        print(f"[完成] 无需更改: {md_path}")            
        
def run_full_scan():
    if not os.path.isdir(DOCS_DIR):
        print(f"[错误] 目录'{DOCS_DIR}'不存在")
        print(f"请将此脚本放在与'docs'目录相同的级别运行。")
        return
    print(f"开始扫描 '{DOCS_DIR}' 目录下的所有 .md 文件...")
    processed = 0
    for root, dirs, files in os.walk(DOCS_DIR):
        for file in files:
            if not file.endswith('.md'): continue
            md_file_path = os.path.join(root, file)
            process_markdown_file(md_file_path)
            processed += 1
    print("\n--- 所有任务完成 ---")
    print(f"总共检查了{processed}个markdown文件")

def run_diff_scan():
    print("正在向 Git 查询已修改(M)或已添加(A)的 .md 文件...")
    try:
        

if __name__ == "__main__":
    main()
                