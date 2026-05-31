import csv,json,sys,os


inpath = r'C:\apple\互联网搜索引擎\test\venv\Scripts\Project2-text_to_image\lora_train\animal\metadata.jsonl'
outpath = r'C:\apple\互联网搜索引擎\test\venv\Scripts\Project2-text_to_image\lora_train\animal\metadata_train.json'
infolder = r'C:\apple\互联网搜索引擎\test\venv\Scripts\Project2-text_to_image\lora_train\animal'

sample_list = []
with open(inpath,) as f:
    for line in f:
        d=json.loads(line)
        print(d)
        sample_list.append({'caption': d['text'], 'file_path': os.path.join(infolder, d['file_name'])})
        
with open(outpath,'w',encoding='utf8') as fw:
    json.dump(sample_list, fw, indent=4)