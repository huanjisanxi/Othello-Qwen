import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import json
import re
from datasets import load_dataset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.env.othello_game import Othello

def parse_coord(coord):
    """Convert 'a1' style coordinate to 0-based indices"""
    if len(coord) < 2:
        return None, None
        
    col, row = coord[0].lower(), coord[1:]
    if not col.isalpha() or not row.isdigit():
        return None, None
        
    col_idx = ord(col) - ord('a')
    row_idx = int(row) - 1
    
    return (row_idx, col_idx) if 0 <= col_idx < 8 and 0 <= row_idx < 8 else (None, None)

def to_coord(row_idx, col_idx):
    """Convert 0-based indices to 'a1' style coordinate"""
    return chr(col_idx + ord('a')) + str(row_idx + 1) if 0 <= row_idx < 8 and 0 <= col_idx < 8 else None

model_path = "/data/data_public/zjy/Othello-Qwen/trainer_output/checkpoint-1000"

data_path = "/data/data_public/zjy/Othello-Qwen/data/training_data_tasks_1_2.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cuda:0" 
)
model.eval()

dataset = load_dataset("json", data_files=data_path)['train']

task1_valid_cnt = 0
task1_invalid_cnt = 0
directions = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]
task2_correct_cnt = 0
task2_wrong_cnt = 0
task2_missed_cnt = 0
with torch.no_grad():
    while True:

        datum = dataset[random.randint(0, len(dataset))]
        input_text = datum['prompt']

        match = re.search(r'Plausible Candidates to Analyze:\n\[(.*?)\]', input_text, re.DOTALL)
        if match:
            candidates_str = match.group(1)
            candidates = [item.strip().strip("'") for item in candidates_str.split(',')]
            
            if len(candidates) > 10:
                processed_candidates = random.sample(candidates, 10)
            else:
                processed_candidates = candidates
            
            processed_str = "['" + "', '".join(processed_candidates) + "']"
            
            input_text = re.sub(r'Plausible Candidates to Analyze:\n\[(.*?)\]', 
                                    f'Plausible Candidates to Analyze:\n{processed_str}', 
                                    input_text, 
                                    flags=re.DOTALL)
        else:
            pass


        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            inputs["input_ids"],
            max_length=2048,
            temperature=0.7
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            end_idx = input_text.index('Analyze a diverse sample of squares to determine which are plausible candidates for a legal move.') 
        except:
            end_idx = input_text.index('Plausible Candidates to Analyze:')
        board_state = json.loads(input_text[input_text.index("Board State:")+len("Board State:"):end_idx].replace("'", '"'))
        board_state['black'], board_state['white'] = board_state['black_pieces'], board_state['white_pieces']
        player = input_text[input_text.index('Player to move: ')+len('Player to move: '):input_text.index('Player to move: ')+len('Player to move: ')+5].lower()
        opponent = "black" if player == "white" else "white"
        try:
            result = json.loads(generated_text[len(input_text):])
        except:
            continue

        if 'Analyze Sampled Squares and Identify Plausible Candidates' in generated_text:
            valid_output, invalid_output = set(), set()
            opponent_pos = board_state['black_pieces'] if opponent == 'black' else board_state['white_pieces']
            for pos in result["final_plausible_candidates"]:
                row, col = parse_coord(pos)

                for dr, dc in directions:
                    nr, nc = row + dr, col + dc

                    if 0<=nr<8 and 0<=nc<8 and to_coord(nr, nc) in opponent_pos:
                            valid_output.add(pos)
                            break
                else:
                    invalid_output.add(pos)
            

            task1_valid_cnt += len(valid_output)
            task1_invalid_cnt += len(invalid_output)
            
            print(f"Task1: valid:{task1_valid_cnt}, invalid:{task1_invalid_cnt}\n")

        elif "Analyze Plausible Candidates for Legality" in generated_text:
            game = Othello()
            game.set_board_state(board_state, player)
            valid_moves = game.get_valid_moves()
            output = result['final_legal_moves']
            actual_moves = set(valid_moves)
            predicted_moves = set(output) 

            correct_predictions = actual_moves & predicted_moves  
            correct_count = len(correct_predictions)

            wrong_predictions = predicted_moves - actual_moves 
            wrong_count = len(wrong_predictions)

            missed_moves = actual_moves - predicted_moves
            missed_count = len(missed_moves)

            print("Task2:")
            print(f"correct {correct_count} 个 ({sorted(correct_predictions)})")
            print(f"wrong: {wrong_count} 个 ({sorted(wrong_predictions)})")
            print(f"missed: {missed_count} 个 ({sorted(missed_moves)})")
            task2_correct_cnt += correct_count
            task2_wrong_cnt += wrong_count
            task2_missed_cnt += missed_count
            print(f"total: correct: {task2_correct_cnt}, wrong: {task2_wrong_cnt}, missed: {task2_missed_cnt}")
            
        else:
            assert False, "invalid output"

        print("-"*100+"\n")