# code to demonstrate that the VAE works well.

import cv2 #tool for image reading and processing
import torch
import numpy as np
from diffusers import AutoencoderKL


model_path = "runwayml/stable-diffusion-v1-5"

VAE = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
# VAE.to("cuda", dtype=torch.float16)

inpath = r'/Users/stevezhou/Desktop/Econ.png'
outpath = r'/Users/stevezhou/Desktop/Econ_VAE.png'

# Use OpenCV to read and adjust image size
raw_image = cv2.imread(inpath)
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) # BGR to RGB
# raw_image = cv2.resize(raw_image, (1024, 1024))

# convert image to np float: tensor -1 ~ 1
image = raw_image.astype(np.float32) / 127.5 - 1.0 

# adjust dimension to macth the form required by Pytorch (N, C, H, W) N: pic(sample) numbers, C: Channel 3 (RGB), H:Height  W:Width
image = image.transpose(2, 0, 1) #It was H * W *C; now it is C * H * W (VAE Convention)
image = image[None, :, :, :] #N: add axis 1 at the beginning

# conver to PyTorch tensor
image = torch.from_numpy(image)  #.to("cuda", dtype=torch.float16) cuda means from cpu to gpu calculation

# Latent
with torch.inference_mode():
    
    latent = VAE.encode(image).latent_dist.sample() 
    print(f"latent's shape is {latent.shape}")
    rec_image = VAE.decode(latent).sample
    

    
    rec_image = (rec_image / 2 + 0.5).clamp(0, 1)
    rec_image = rec_image.cpu().permute(0, 2, 3, 1).numpy() #dimension is converted back to the original one of picture

   
    rec_image = (rec_image * 255).round().astype("uint8") 
    rec_image = rec_image[0] 

  
    cv2.imwrite(outpath, cv2.cvtColor(rec_image, cv2.COLOR_RGB2BGR))