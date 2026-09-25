# for testing/evaluating LoRA

import argparse
import os
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont


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
parser.add_argument("--prompt", nargs="+", default=["Minimalist sculpture of a vibrant yellow parrot perched on a sleek black stand"],
                    help="one or more prompts")
parser.add_argument("--compare", action="store_true",
                    help="also render each prompt without the LoRA (same seed) and save the pair side by side")
parser.add_argument("--negative_prompt", default="blurry, low quality, deformed, distorted face, extra limbs, text, watermark")
parser.add_argument("--num_inference_steps", type=int, default=30)
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--hires_scale", type=float, default=1.0,
                    help="e.g. 1.5: upscale the 512px result and refine it with img2img for sharper detail (1.0 = off)")
parser.add_argument("--hires_strength", type=float, default=0.45, help="how much the hires pass may change the image")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", default="test_generations")
args = parser.parse_args()

device = get_device()


test_pipe = StableDiffusionPipeline.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None
).to(device)
# DPM-Solver++ gives sharper results than the default PNDM scheduler at ~30 steps
test_pipe.scheduler = DPMSolverMultistepScheduler.from_config(test_pipe.scheduler.config)


unet = test_pipe.unet
unet = PeftModel.from_pretrained(unet, args.lora_path)
test_pipe.unet = unet
# shares every component (including the LoRA UNet) with test_pipe, so no extra memory
hires_pipe = StableDiffusionImg2ImgPipeline(**test_pipe.components) if args.hires_scale > 1 else None


os.makedirs(args.output_dir, exist_ok=True)


def generate(prompt):
    image = test_pipe(
        prompt,
        negative_prompt=args.negative_prompt or None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed)
    ).images[0]
    if hires_pipe is not None:
        size = int(image.width * args.hires_scale) // 8 * 8
        image = hires_pipe(
            prompt,
            image=image.resize((size, size), Image.LANCZOS),
            strength=args.hires_strength,
            negative_prompt=args.negative_prompt or None,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator("cpu").manual_seed(args.seed)
        ).images[0]
    return image


def side_by_side(left, right, labels, header=48):
    """Put two images next to each other with a label above each."""
    canvas = Image.new("RGB", (left.width + right.width, left.height + header), "white")
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width, header))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)  # common on Linux
    except OSError:
        try:
            font = ImageFont.load_default(size=28)
        except TypeError:  # Pillow < 10.1 only has a tiny bitmap font
            font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for x, label in [(0, labels[0]), (left.width, labels[1])]:
        draw.text((x + 12, 10), label, fill="black", font=font)
    return canvas


for i, prompt in enumerate(args.prompt):
    image = generate(prompt)
    name = Path(args.lora_path).name + (f"_{i+1}" if len(args.prompt) > 1 else "")

    if args.compare:
        with unet.disable_adapter():
            base_image = generate(prompt)
        image = side_by_side(base_image, image, ["SD1.5 base", "+ LoRA"])
        output_path = os.path.join(args.output_dir, f"compare_{name}.jpg")
    else:
        output_path = os.path.join(args.output_dir, f"generation_{name}.jpg")

    image.save(output_path)
    print(f"✅ Test Images are saved: {output_path}  ({prompt})")
