import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

import yaml
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoConfig,
)
from trl import SFTTrainer, SFTConfig
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def train_model(config: dict):
    resume_checkpoint = config['training_params'].get('resume_from_checkpoint')  # 从配置获取checkpoint路径
    model_id = config['model_params']['model_id']

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        load_path = resume_checkpoint 
    else:
        print(f"Loading base model: {model_id}")
        load_path = model_id  

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model_params']['use_4bit'],
        bnb_4bit_quant_type=config['model_params']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, config['model_params']['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=True  
    )

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        quantization_config=bnb_config,
        device_map=config['device'],
        trust_remote_code=True,
        # attn_implementation="flash_attention_2"
    )
    
    print(f"Loading data from: {config['data_params']['dataset_path']}")
    dataset_dict = load_dataset("json", data_files=config['data_params']['dataset_path'])
    dataset = dataset_dict['train']

    peft_config = LoraConfig(** config['lora_params'])

    training_args = SFTConfig(
        max_length=config['training_params']['max_length'],
        per_device_train_batch_size=config['training_params']['batch_size'],
        dataset_kwargs={"format": "prompt-completion"},
    )
    
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )
    
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    final_output_dir = f"{config['training_params']['output_dir']}/final_checkpoint"
    print(f"Training finished. Saving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    print("Done.")


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