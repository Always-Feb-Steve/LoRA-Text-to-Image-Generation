import os,sys,json,argparse

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

parser = argparse.ArgumentParser(description="Build captions.json from a folder of <tag>/ image sub-folders")
parser.add_argument('--infolder', default='lora_train', help="each sub-folder name is used as the caption tag")
parser.add_argument('--outpath', default='lora_train/captions.json')
parser.add_argument('--tags', nargs='*', default=None, help="only include these sub-folders (default: all)")
args = parser.parse_args()


sample_list = []
for tag in sorted(os.listdir(args.infolder)):
    one_folder = os.path.join(args.infolder, tag)
    if not os.path.isdir(one_folder): continue
    if args.tags and tag not in args.tags: continue
    for one_img_path in sorted(os.listdir(one_folder)):
        if not one_img_path.lower().endswith(IMAGE_EXTS): continue
        one_img_path = os.path.join(one_folder, one_img_path)
        sample_list.append({'caption': f"a photo of '{tag}'", 'file_path': one_img_path})

with open(args.outpath,'w',encoding='utf8') as fw:
    json.dump(sample_list, fw, indent=4)

print(f"Wrote {len(sample_list)} samples to {args.outpath}")
