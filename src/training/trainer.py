import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

from src.data_process.cot_loader import load_cot_data

def train_model(config: dict):
    print(f"Loading base model: {config['model_params']['model_id']}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model_params']['use_4bit'],
        bnb_4bit_quant_type=config['model_params']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, config['model_params']['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=True  
    )

    model = AutoModelForCausalLM.from_pretrained(
        config['model_params']['model_id'],
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    
    print(f"Loading data from: {config['data_params']['dataset_path']}")
    training_strings = load_cot_data(config['data_params']['dataset_path'])
    
    dataset = Dataset.from_dict({"text": training_strings})
    train_dataset = dataset

    # print("Configuring LoRA...")
    # model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(** config['lora_params'])
    # model = get_peft_model(model, peft_config)

    training_args = SFTConfig(
        max_length=config['training_params']['max_length'],
        # assistant_only_loss=True,
    )
    
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config
    )
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    
    print("Starting training...")
    trainer.train()
    
    final_output_dir = f"{config['training_params']['output_dir']}/final_checkpoint"
    print(f"Training finished. Saving final model to {final_output_dir}...")
    trainer.save_model(final_output_dir)
    print("Done.")