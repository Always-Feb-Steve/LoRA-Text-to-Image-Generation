# for testing/evaluating LoRA

import torch, os, json
from pathlib import Path
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, PeftModel



model_name = "runwayml/stable-diffusion-v1-5"
epoch = 40
lora_path = rf"C:\apple\互联网搜索引擎\test\venv\Scripts\Project2-text_to_image\models\lora_trained_animal\checkpoints\checkpoint_epoch_{epoch}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None
).to(device)


unet = test_pipe.unet
unet = PeftModel.from_pretrained(unet, lora_path)
test_pipe.unet = unet


test_prompt = "Minimalist sculpture of a vibrant yellow parrot perched on a sleek black stand"


test_output_dir = "test_generations"
os.makedirs(test_output_dir, exist_ok=True)


image = test_pipe(
    test_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]


output_path = os.path.join(test_output_dir, f"generation_epoch_{epoch}.jpg")
image.save(output_path)

print(f"✅ Test Images are saved: {output_path}")
