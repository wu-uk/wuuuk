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
        cmd = ['git', '-c', 'core.quotepath=false', 'diff', 'HEAD', '--name-only', '--diff-filter=AM']
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', check=True)
        changed_files = result.stdout.strip().splitlines()
        if not changed_files:
            print("Git 报告：没有找到自上次 commit 以来修改过的文件。")
            print("如果你是想对 *未暂存*(unstaged)的文件生效，请先 'git add .'")
            return
        print(f"Git 找到 {len(changed_files)} 个变更文件。开始过滤 .md ...")
        processed = 0
        for file_path_str in changed_files:
            print(file_path_str)
            # "docs/course/ComputerNetwork/\350\256\241\347\256\227\346\234\272\347\275\221\347\273\234\347\254\254\345\205\255\346\254\241\344\275\234\344\270\232.md"
            file_path = file_path_str.strip('"')
            if file_path.startswith('docs') and file_path.endswith('.md'):
                # file_path = DOCS_DIR + file_path.lstrip('docs')
                file_path = file_path.replace('/', '\\')
                print(file_path)
                # D:\Project\NewBlog\docs\course\ComputerNetwork\\350\256\241\347\256\227\346\234\272\347\275\221\347\273\234\347\254\254\345\233\233\346\254\241\344\275\234\344\270\232.md
                if os.path.exists(file_path):
                    process_markdown_file(file_path)
                    processed += 1
        if processed == 0:
            print(f"在 {len(changed_files)} 个变更文件中，没有找到 'docs/' 目录下的 .md 文件。")
        
    except FileNotFoundError:
        print("错误: 'git' 命令未找到。")
        print("请确保你已安装 Git 并且它在你的系统 PATH 中。")
        sys.exit(1)
        
    except subprocess.CalledProcessError as e:
        print(f"Git 命令执行失败 (Error code {e.returncode}):")
        if e.stderr: print(e.stderr)
        if e.stdout: print(e.stdout)
        print("\n这通常发生在你 *还未进行过任何 commit* 的新仓库中。")
        print("请先执行 'git add .' 和 'git commit -m \"initial commit\"'。")
        sys.exit(1)
    except Exception as e:
        print(f"发生未知错误: {e}")
        sys.exit(1)

    print("\n--- 增量任务完成 ---")
        
def main():
    parser = argparse.ArgumentParser(
        description="自动修复 MkDocs 中的 Markdown 图片链接（下载或复制）。",
        epilog="默认行为 (不带参数): 仅扫描 'git diff' 报告的已修改文件。"
    )
    parser.add_argument(
        '-f', '--full',
        action='store_true',  # 这意味着，如果参数存在，就设为 True
        help="运行全量扫描，检查 'docs/' 下的 *所有* .md 文件，而不仅仅是 Git 变更文件。"
    )
    args = parser.parse_args()
    if not os.path.isdir(DOCS_DIR):
        print(f"[错误] 目录'{DOCS_DIR}'不存在")
        print(f"请将此脚本放在与'docs'目录相同的级别运行。")
        return 
    if args.full:
        run_full_scan()
    else:
        run_diff_scan()

if __name__ == "__main__":
    main()
                