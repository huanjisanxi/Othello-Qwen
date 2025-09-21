import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_process.cot_loader import load_cot_data

model_path = "/data/data_public/zjy/Othello-Qwen/trainer_output/checkpoint-5000"

data_path = "/data/data_public/zjy/Othello-Qwen/data/othello_with_cot.json"

data_strings = load_cot_data(data_path, 'eval')

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cuda:0" 
)

input_text = data_strings[random.randint(1, 10000)]

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    inputs["input_ids"],
    max_length=1024,
    temperature=0.7
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)