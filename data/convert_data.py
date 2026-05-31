import os,sys,json

infolder = r'C:\apple\互联网搜索引擎\test\venv\Scripts\Project2-text_to_image\lora_train'
outpath = r'./captions.json'



sample_list = []
for tag in os.listdir(infolder):
    if tag != 'Spirited_Away': continue
    one_folder = os.path.join(infolder, tag)
    for one_img_path in os.listdir(one_folder):
        one_img_path = os.path.join(one_folder, one_img_path)
        sample_list.append({'caption': f"a photo of '{tag}'", 'file_path': one_img_path})
        
with open(outpath,'w',encoding='utf8') as fw:
    json.dump(sample_list, fw, indent=4)