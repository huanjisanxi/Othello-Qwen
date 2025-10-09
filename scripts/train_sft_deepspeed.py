# scripts/train_sft_deepspeed.py

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset
import yaml

def train_model(config: dict):
    # DeepSpeed会自动初始化分布式环境
    
    resume_checkpoint = config['training_params'].get('resume_from_checkpoint') 
    model_id = config['model_params']['model_id']

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        load_path = resume_checkpoint 
    else:
        print(f"Loading base model: {model_id}")
        load_path = model_id  

    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model_params']['use_4bit'],
        bnb_4bit_quant_type=config['model_params']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, config['model_params']['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=True  
    )

    # 对于DeepSpeed，让DeepSpeed管理设备分配
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        quantization_config=bnb_config,
        device_map=None,  # 让DeepSpeed处理设备分配
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据
    dataset_dict = load_dataset("json", data_files=config['data_params']['dataset_path'])
    dataset = dataset_dict['train']

    # LoRA配置
    peft_config = LoraConfig(**config['lora_params'])

    # 训练参数 - 包含DeepSpeed配置
    training_args = SFTConfig(
        max_length=config['training_params']['max_length'],
        per_device_train_batch_size=config['training_params']['batch_size'],
        dataset_kwargs={"format": "prompt-completion"},
        # deepspeed="./config/ds_config.json",
        
        # # 训练参数
        # bf16=True,
        # gradient_checkpointing=True,
        # logging_steps=10,
        # save_steps=500,
        # output_dir=config['training_params']['output_dir'],
        # learning_rate=config['training_params'].get('learning_rate', 2e-4),
        # num_train_epochs=config['training_params'].get('num_epochs', 3),
        # warmup_ratio=0.03,
        # save_total_limit=3,
        # dataloader_drop_last=True,
    )
    
    print("Initializing SFTTrainer with DeepSpeed...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )
    
    print("Starting DeepSpeed training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # 保存模型
    final_output_dir = f"{config['training_params']['output_dir']}/final_checkpoint"
    print(f"Saving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    print("Training completed!")

def main():
    config_path = "config/default.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    train_model(config)

if __name__ == "__main__":
    main()