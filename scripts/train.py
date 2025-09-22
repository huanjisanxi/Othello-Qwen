import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

import yaml
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_process.cot_loader import load_cot_data, convert_to_dataset_dict

def train_model(config: dict):
    model_id = config['model_params']['model_id']
    print(f"Loading base model: {model_id}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model_params']['use_4bit'],
        bnb_4bit_quant_type=config['model_params']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, config['model_params']['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=True  
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=config['device'],
        trust_remote_code=True,
    )
    
    print(f"Loading data from: {config['data_params']['dataset_path']}")
    training_strings = load_cot_data(config['data_params']['dataset_path'])
    negative_strings = load_cot_data(config['data_params']['dataset_path_negative'])
    all_strings = training_strings + negative_strings

    dataset = Dataset.from_dict(convert_to_dataset_dict(all_strings))

    peft_config = LoraConfig(** config['lora_params'])

    training_args = SFTConfig(
        max_length=config['training_params']['max_length'],
        per_device_train_batch_size=config['training_params']['batch_size']
    )
    
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config
    )
    
    print("Starting training...")
    trainer.train()
    
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