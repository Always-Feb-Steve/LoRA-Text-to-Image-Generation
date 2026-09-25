# simple_lora_trainer.py
#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# for training LoRA

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json 
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model 
import numpy as np


def get_device():
    """优先使用 CUDA，其次 Apple Silicon 的 MPS，最后 CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LoraDataset(Dataset):
    """LoRA训练数据集"""
    
    def __init__(self, captions_file, transform=None, image_size=512):
        self.transform = transform
        self.image_size = image_size
        
        # 加载标签数据
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 统计
        print(f"加载了 {len(self.data)} 张图片")
        
        # 如果没有提供transform，创建默认的
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 读取图片（支持中文路径）
        img_path = item["file_path"]
        
        try:
            # 方法1：使用PIL直接打开
            image = Image.open(img_path).convert("RGB")
        except:
            # 方法2：如果失败，使用opencv
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is not None:
                    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    # 如果还是失败，创建空白图片
                    print(f"警告: 无法读取图片 {img_path}")
                    image = Image.new('RGB', (self.image_size, self.image_size), color='white')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), color='white')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签文本
        caption = item["caption"]
        
        return {
            "pixel_values": image,
            "caption": caption
        }






class SimpleLoraTrainer:
    """简化的LoRA训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        
        # 创建输出目录
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        torch.manual_seed(config.get("seed", 42))
        
        # VAE缩放因子（Stable Diffusion 1.x 使用 0.18215）
        # self.vae_scale_factor = 0.18215
    
    def prepare_models(self):
        """准备模型"""
        print("准备模型...")
        
        model_id = self.config["model_name"]
        
        # 0. 加载VAE模型
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae"
        ).to(self.device)
        
        # 冻结VAE
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # 1. 加载文本编码器和分词器 #得到prompt的toen
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer"
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder"
        ).to(self.device)
        
        # 冻结文本编码器
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # 2. 加载UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet"
        ).to(self.device)
        
        
        # 3. 配置LoRA
        lora_config = LoraConfig(
            r=self.config.get("lora_rank", 4),  # LoRA秩
            lora_alpha=self.config.get("lora_alpha", 4),
            target_modules=["to_k", "to_q", "to_v", "to_out.0", 
                            "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"], #文生图就训这几个parameters
            lora_dropout=self.config.get("lora_dropout", 0.0),
            bias="none"
        )
        
        # 应用LoRA
        self.unet = get_peft_model(self.unet, lora_config) #插入LoRA
        self.unet.print_trainable_parameters()
        
        # 4. 加载噪声调度器
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler"
        )
        
        # 5. 准备优化器
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=1e-2
        )
        
        print(f"✅ 模型准备完成，使用设备: {self.device}")
    
    def prepare_data(self):
        """准备数据"""
        print("准备数据...")
        
        dataset = LoraDataset( #加载图片
            captions_file=self.config["captions_file"],
            image_size=self.config.get("image_size", 512)
        )
        
        use_cuda = self.device.type == "cuda"
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=2 if use_cuda else 0,  # MPS/CPU 上多进程加载反而更慢
            pin_memory=use_cuda
        )
        
        return dataloader
    
    def encode_images(self, images):
        """将图像编码到潜在空间
        
        Args:
            images: 形状为 [B, 3, H, W] 的张量，值在 [-1, 1] 范围
        
        Returns:
            latents: 形状为 [B, 4, H/8, W/8] 的潜在表示
        """
        with torch.no_grad():
            # VAE编码
            latents = self.vae.encode(images).latent_dist.sample()
            # 缩放潜在表示
            latents = latents * self.vae.config.scaling_factor #别人用unet就是在这个scaling factor训的，为了保持unet原来的能力所以要这个scaling factor
        return latents
    
    def tokenize_captions(self, captions):
        """分词标签文本"""
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids.to(self.device) #切词
    
    def train_step(self, batch):
        """单个训练步骤"""
        # 获取数据
        images = batch["pixel_values"].to(self.device)
        captions = batch["caption"]
        
        # 编码文本
        input_ids = self.tokenize_captions(captions)
        with torch.no_grad(): #不计算梯度
            encoder_hidden_states = self.text_encoder(input_ids)[0]
        
        # 编码图像到潜在空间
        latents = self.encode_images(images)  # [N, 4, H/8, W/8]
        
        # 采样噪声和时间步
        noise = torch.randn_like(latents) 
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        # 加噪
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 预测噪声是怎么来的，以此来反推来remove noise
        noise_pred = self.unet( 
            noisy_latents,
            timesteps,
            encoder_hidden_states
        ).sample
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def train(self):
        """训练主循环"""
        print("开始训练...")
        
        # 准备模型和数据
        self.prepare_models()
        dataloader = self.prepare_data()
        
        # 训练参数
        num_epochs = self.config.get("num_epochs", 10)
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        
        # 训练循环
        global_step = 0
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*60)
            
            self.unet.train() #告诉模型要开始train了，dropout会启动
            epoch_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(progress_bar): 
                # 前向传播
                loss = self.train_step(batch)
                
                # 缩放损失（如果使用梯度累积）
                loss = loss / gradient_accumulation_steps
                
                # 反向传播: 
                loss.backward() #partial differentiation: where to go, the magnitude
                
                # 梯度累积
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 0.25)
                    
                    # 更新参数
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # 更新进度条
                epoch_loss += loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                global_step += 1
                
                # 保存检查点 每20步保存一个模型，最后再来评测他们
                if global_step % self.config.get("save_steps", 100) == 0:
                    self.save_checkpoint(global_step)
            
            # 每个epoch结束后保存
            avg_loss = epoch_loss / len(dataloader) 
            print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
            
            self.save_checkpoint(f"epoch_{epoch+1}")
        
        # 训练完成
        self.save_final_model()
        print("\n✅ 训练完成!")
    
    def save_checkpoint(self, step):
        """保存检查点"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 保存LoRA权重
        checkpoint_path = checkpoint_dir / f"checkpoint_{step}"
        self.unet.save_pretrained(checkpoint_path)
        
        # 保存优化器状态
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_path / "optimizer.pt"
        )
        
        print(f"✅ 检查点已保存: {checkpoint_path}")
    
    def save_final_model(self):
        """保存最终模型"""
        print("\n保存最终模型...")
        
        # 保存LoRA权重
        self.unet.save_pretrained(self.output_dir)
        
        # 保存配置文件
        config = {
            "model_type": "lora",
            "base_model": self.config["model_name"],
            "lora_config": {
                "r": self.config.get("lora_rank", 4),
                "alpha": self.config.get("lora_alpha", 4),
                "dropout": self.config.get("lora_dropout", 0.0)
            },
            "training_config": self.config
        }
        
        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 最终模型已保存到: {self.output_dir}")

def create_default_config():
    """创建默认配置"""
    return {
        "model_name": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "captions_file": "lora_train/captions.json", #输入的prompt：tag和图片路径
        "output_dir": "./models/lora_trained", #输出一个lora的模型
        
        # 训练参数
        "learning_rate": 5e-6,
        "batch_size": 2,  # 小批量，减少显存占用
        "gradient_accumulation_steps": 2,  # 累积梯度
        "num_epochs": 100,
        "image_size": 512,
        
        # LoRA参数
        "lora_rank": 4,  # M*N = M*k * k*N
        "lora_alpha": 4, 
        "lora_dropout": 0.0, #这里都得学，dropout不用学
        
        # 保存和验证
        "save_steps": 20,
        
        # 其他
        "seed": 42 
    }





def parse_args():
    """命令行参数，覆盖默认配置中的对应项"""
    config = create_default_config()
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on Stable Diffusion 1.5")
    for key in ["model_name", "captions_file", "output_dir"]:
        parser.add_argument(f"--{key}", default=config[key])
    for key in ["batch_size", "gradient_accumulation_steps", "num_epochs", "image_size",
                "lora_rank", "lora_alpha", "save_steps", "seed"]:
        parser.add_argument(f"--{key}", type=int, default=config[key])
    parser.add_argument("--learning_rate", type=float, default=config["learning_rate"])
    config.update(vars(parser.parse_args()))
    return config


def main():
    """主函数"""
    print("="*60)
    print("简化版LoRA训练器")
    print("="*60)

    # 加载配置
    config = parse_args()

    # 检查数据是否存在
    if not Path(config["captions_file"]).exists():
        print(f"❌ 标签文件不存在: {config['captions_file']}")
        print("请先运行 data/convert_data.py 或 data/convert_data_from_jsonl.py 准备数据")
        return
    
    # 创建训练器
    trainer = SimpleLoraTrainer(config)
    
    # 开始训练
    trainer.train()
    
    # 生成使用示例
    print("\n" + "="*60)
    print("使用训练好的LoRA:")
    print("="*60)
    print(f"python eval_LoRA.py --lora_path {config['output_dir']} --prompt \"<your prompt>\"")


if __name__ == "__main__":
    main()