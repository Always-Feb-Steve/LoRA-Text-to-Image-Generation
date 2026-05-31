# for testing/evaluating LoRA

import torch, os, json
from pathlib import Path
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, PeftModel

# 加载基础模型

model_name = "runwayml/stable-diffusion-v1-5"
# model_name = r"C:\apple\互联网搜索引擎\test\venv\Scripts\sd1.5"
epoch = 40
lora_path = rf"C:\apple\互联网搜索引擎\test\venv\Scripts\Project2-text_to_image\models\lora_trained_animal\checkpoints\checkpoint_epoch_{epoch}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None
).to(device)

# 加载当前训练的LoRA权重
# 注意：这里需要保存并重新加载LoRA权重
# 加载配置信息
unet = test_pipe.unet
unet = PeftModel.from_pretrained(unet, lora_path)
test_pipe.unet = unet

# 生成测试图片
test_prompt = "Minimalist sculpture of a vibrant yellow parrot perched on a sleek black stand"

# 创建输出目录
test_output_dir = "test_generations"
os.makedirs(test_output_dir, exist_ok=True)

# 生成图片
image = test_pipe(
    test_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

# 保存图片
output_path = os.path.join(test_output_dir, f"generation_epoch_{epoch}.jpg")
image.save(output_path)

print(f"✅ 测试图片已保存: {output_path}")