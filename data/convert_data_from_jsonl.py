import csv,json,sys,os,argparse

# metadata.jsonl uses the Hugging Face imagefolder format: {"file_name": "xxx.jpg", "text": "caption"}
parser = argparse.ArgumentParser(description="Convert <infolder>/metadata.jsonl into captions.json")
parser.add_argument('--infolder', default='lora_train/animal', help="folder containing the images and metadata.jsonl")
parser.add_argument('--inpath', default=None, help="defaults to <infolder>/metadata.jsonl")
parser.add_argument('--outpath', default='lora_train/captions.json')
args = parser.parse_args()

inpath = args.inpath or os.path.join(args.infolder, 'metadata.jsonl')

sample_list = []
with open(inpath, encoding='utf8') as f:
    for line in f:
        if not line.strip(): continue
        d=json.loads(line)
        sample_list.append({'caption': d['text'], 'file_path': os.path.join(args.infolder, d['file_name'])})

with open(args.outpath,'w',encoding='utf8') as fw:
    json.dump(sample_list, fw, indent=4)

print(f"Wrote {len(sample_list)} samples to {args.outpath}")
