# This code is for generating image (inference); I use a pre-trained Unet
import argparse
import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

#pipeline is a wrapper including: text encoder, UNet, VAE, Scheduler, Tokenizer, and Safety
# prompt -> tokenizer: converts text into a list of token ID -> Text Encoder: Takes those token IDs and converts them into semantic embeddings (vectors) ->Unet

OUTPUT_DIR = "generated_images"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device):
    # half precision only on CUDA; MPS/CPU run in float32
    return torch.float16 if device == "cuda" else torch.float32


def save_images(images, prefix):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for idx, img in enumerate(images):
        file_path = os.path.join(OUTPUT_DIR, f"{prefix}_{idx+1}.png")
        img.save(file_path)
        print(f"Saved {file_path}")


def generate_and_show_images(pipeline, prompt, negative_prompt=None, num_images=3, save_prefix=None):

    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=50,  # how many denosing steps the model takes to generate the image
                                # trade off: fewer steps lead to faster generation but less detailed pictures; more steps lead to slower genration but more detailed pictures
        guidance_scale=7.5,
        generator=torch.Generator("cpu").manual_seed(42) # fixed seed so results are reproducible on any device
    ).images

    if save_prefix:
        save_images(images, save_prefix)

    return images


def run_sd1_5(prompt, model_name="stable-diffusion-v1-5/stable-diffusion-v1-5", device=None):
    device = device or get_device()
    print(f"Loading SD1.5 model: {model_name}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=get_dtype(device),
        safety_checker=None,
    ).to(device)


    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()


    print(f"Generating image with prompt: {prompt}")
    images = generate_and_show_images(
        pipe,
        prompt=prompt,
        negative_prompt="low quality, blurry, deformed, ugly",
        num_images=3,
        save_prefix="sd1_5_output"
    )

    return images



def run_sdxl(prompt, model_name="stabilityai/stable-diffusion-xl-base-1.0", device=None, use_refiner=True):
    """
    Use Stable Diffusion XL to generate images
    """
    device = device or get_device()
    dtype = get_dtype(device)
    print(f"Loading SDXL model: {model_name}")


    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    ).to(device)


    refiner = None
    if use_refiner:
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=dtype,
                variant="fp16",
                use_safetensors=True,
            ).to(device)
        except Exception as e:
            print(f"Could not load refiner: {e}")


    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()


    print(f"Generating image with SDXL for prompt: {prompt}")


    images = pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry, deformed",
        num_inference_steps=50,  # SDXL needs more steps
        denoising_end=0.8 if refiner else None,  # swtich to refiner at 80%
        output_type="latent" if refiner else "pil",
        guidance_scale=7.5,
        generator=torch.Generator("cpu").manual_seed(42)
    ).images


    if refiner:
        images = refiner(
            prompt=prompt,
            image=images,
            num_inference_steps=50,
            denoising_start=0.8,
            guidance_scale=7.5,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images

    save_images(images, "sdxl_output")

    return images


def run_sd3(prompt, model_name="stabilityai/stable-diffusion-3-medium-diffusers", device=None):
    # SD3 is a gated model: accept the license on Hugging Face and run `huggingface-cli login` first
    device = device or get_device()
    print(f"Loading SD3 model: {model_name}")


    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_name,
        torch_dtype=get_dtype(device),
    ).to(device)


    pipe.enable_attention_slicing()


    images = pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry",
        num_inference_steps=28,
        guidance_scale=7.0,
        width=1024,
        height=1024,
        generator=torch.Generator("cpu").manual_seed(42)
    ).images


    save_images(images, "sd3_output")
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline text-to-image inference (no LoRA)")
    parser.add_argument("--model", choices=["sd15", "sdxl", "sd3"], default="sd15")
    parser.add_argument("--model_name", default=None, help="override the Hugging Face model id or a local path")
    parser.add_argument("--prompt", default="LeBron James in Golden State Warrior's jersey dunks , realistic art, detailed, 4k")
    parser.add_argument("--no_refiner", action="store_true", help="SDXL only: skip the refiner to save memory")
    args = parser.parse_args()

    kwargs = {"model_name": args.model_name} if args.model_name else {}
    if args.model == "sd15":
        run_sd1_5(args.prompt, **kwargs)
    elif args.model == "sdxl":
        run_sdxl(args.prompt, use_refiner=not args.no_refiner, **kwargs)
    else:
        run_sd3(args.prompt, **kwargs)
