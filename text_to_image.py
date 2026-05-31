# This code is for generating image (inference); I use a pre-trained Unet
import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers import FluxPipeline
import matplotlib.pyplot as plt
from PIL import Image

#pipeline is a wrapper including: text encoder, UNet, VAE, Scheduler, Tokenizer, and Safety 
# prompt -> tokenizer: converts text into a list of token ID -> Text Encoder: Takes those token IDs and converts them into semantic embeddings (vectors) ->Unet


def generate_and_show_images(pipeline, prompt, negative_prompt=None, num_images=3, save_path=None):
    
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=50,  # how many denosing steps the model takes to generate the image
                                # trade off: fewer steps lead to faster generation but less detailed pictures; more steps lead to slower genration but more detailed pictures
        guidance_scale=7.5,   
        generator=torch.Generator(device="cuda").manual_seed(42) if torch.cuda.is_available() else None # use cpu or gpu
    ).images
   
    # save images-formulaic
    if save_path:
        for idx, img in enumerate (images):
            img.save(f"{save_path}_{idx}.png")
            
    return images


def run_sd1_5(prompt, model_name="runwayml/stable-diffusion-v1-5", device="cpu"):
    print(f"Loading SD1.5 model: {model_name}")
    # 加载管道
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,  # 可选：禁用安全检查器以加快速度
    ).to(device)
   
    # 启用内存优化（可选）
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
   
    # 生成图像
    print(f"Generating image with prompt: {prompt}")
    images = generate_and_show_images(
        pipe,
        prompt=prompt,
        negative_prompt="low quality, blurry, deformed, ugly",  
        num_images=3,
        save_path="sd1_5_output"
    )
   
    return images



def run_sdxl(prompt, model_name="stabilityai/stable-diffusion-xl-base-1.0", device="cpu"):
    """
    SUe Stable Diffusion XL to generate images
    """
    print(f"Loading SDXL model: {model_name}")
   
    # SDXL 需要两个模型：base 和 refiner
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant="fp16",
        use_safetensors=True,
    ).to(device)
   
    # 加载refiner模型（可选但推荐）
    refiner = None
    try:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16",
            use_safetensors=True,
        ).to(device)
    except Exception as e:
        print(f"Could not load refiner: {e}")
   
    # 优化
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
   
    # 生成图像
    print(f"Generating image with SDXL for prompt: {prompt}")
   
    # 第一步：使用base模型生成
    images = pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry, deformed",
        num_inference_steps=50,  # SDXL needs more steps
        denoising_end=0.8,  # swtich to refiner at 80%
        output_type="latent" if refiner else "pil",
        guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(42)
    ).images
   
    # 第二步：使用refiner精炼
    if refiner:
        images = refiner(
            prompt=prompt,
            image=images,
            num_inference_steps=50,
            denoising_start=0.8,
            guidance_scale=7.5,
            generator=torch.Generator(device=device).manual_seed(42)
        ).images
    
    # 保存图像
    folder_name="generated_images"
    os.makedirs(folder_name, exist_ok=True)

    for idx, img in enumerate(images):
        file_path=os.path.join(folder_name,f"sdxl_output_{idx+1}.png")
        img.save(file_path)
   
    return images



# Using the Model SD 1.5 or XL

# # model_path = r'./sd1.5'
# # model_path = "runwayml/stable-diffusion-v1-5"
# # 使用示例
# sd1_5_images = run_sd1_5(
#     prompt="LeBron James in LA Laker jersey dunks , realistic art, detailed, 4k",
#     model_name="runwayml/stable-diffusion-v1-5",  # 其他可选模型:
#     # "CompVis/stable-diffusion-v1-4"
#     # "dreamlike-art/dreamlike-diffusion-1.0"
#     # "prompthero/openjourney"
# )


# 使用示例
sdxl_images = run_sdxl(
    prompt="LeBron James in Golden State Warrior's jersey dunks , realistic art, detailed, 4k",
    model_name="stabilityai/stable-diffusion-xl-base-1.0"
)

def run_sd3(prompt, model_name="stabilityai/stable-diffusion-3-medium-diffusers", device="cpu"):
    """
    使用 Stable Diffusion 3 生成图像
    注意：SD3需要特定版本的diffusers和transformers
    """
    print(f"Loading SD3 model: {model_name}")
   
    # SD3需要特定的导入方式
    from diffusers import StableDiffusion3Pipeline
   
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
   
    # 优化设置
    pipe.enable_attention_slicing()
   
    # SD3特定的参数
    images = pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry",
        num_inference_steps=28,
        guidance_scale=7.0,
        width=1024,  # SD3支持更高分辨率
        height=1024,
        generator=torch.Generator(device=device).manual_seed(42)
    ).images
   
    # 显示和保存
    for idx, img in enumerate(images):
        img.save(f"sd3_output_{idx+1}.png")
    return images
       
# # 使用示例（注意：SD3模型较大，需要足够显存）
# model_path = r'./models/stabilityai_stable-diffusion-3-medium-diffusers/stabilityai/stable-diffusion-3-medium-diffusers'
# # model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
# sd3_images = run_sd3(
#     prompt="An astronaut riding a horse on Mars, cinematic, hyperrealistic",
#     model_name=model_path
# )