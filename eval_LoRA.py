# for testing/evaluating LoRA

import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


parser = argparse.ArgumentParser(description="Generate a test image from a trained LoRA checkpoint")
parser.add_argument("--model_name", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
parser.add_argument("--lora_path", default="models/lora_trained",
                    help="final model dir, or a checkpoint such as models/lora_trained/checkpoints/checkpoint_epoch_40")
parser.add_argument("--prompt", default="Minimalist sculpture of a vibrant yellow parrot perched on a sleek black stand")
parser.add_argument("--num_inference_steps", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", default="test_generations")
args = parser.parse_args()

device = get_device()


test_pipe = StableDiffusionPipeline.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None
).to(device)


unet = test_pipe.unet
unet = PeftModel.from_pretrained(unet, args.lora_path)
test_pipe.unet = unet


os.makedirs(args.output_dir, exist_ok=True)


image = test_pipe(
    args.prompt,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=7.5,
    generator=torch.Generator("cpu").manual_seed(args.seed)
).images[0]


output_path = os.path.join(args.output_dir, f"generation_{Path(args.lora_path).name}.jpg")
image.save(output_path)

print(f"✅ Test Images are saved: {output_path}")
