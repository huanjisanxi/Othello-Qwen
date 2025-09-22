import json
import random
import sys
import os
from pathlib import Path
from copy import deepcopy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.othello_utils import(
    coord_to_position,
    check_captures_from_string,
    convert_capture_to_english_cot
)

file_path = '/data/data_public/zjy/Othello-Qwen/data/othello_with_cot.json'
with open(file_path, 'r') as f:
    data = json.load(f)

def random_chess_position():
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows = range(1, 9)
    
    col = random.choice(columns)
    row = random.choice(rows)
    
    return f"{col}{row}"

negative_data = []

data_cnt = 0
OCCUPIED_NUM = 1500
NO_FLANK_NUM = 1500


def sample_steps_occupied(record):
    game = random.choice(record)
    analysis = game['cot_analysis'][:40]
    ret_game = deepcopy(game)
    del ret_game['cot_analysis']

    data_cnt = 0
    for step in analysis:
        if step['cot_format'] != 'valid_json':
            continue

        data_cnt += 1
        board_state_split = step['board_state'].replace(' ', '').split('\n')
        row, col = random.randint(0, 7), random.randint(0, 7)
        while board_state_split[row][col] == '.':
            row, col = random.randint(0, 7), random.randint(0, 7)
        occupied_by = board_state_split[row][col]
        occupied_by = 'Black' if occupied_by == 1 else 'White'
        pos = coord_to_position(row, col)

        step['position'] = pos
        step['cot'] = {
            "analysis": {
                "chosen_move": pos,
                "is_legal": False,
                "error_type": "Position Occupied",
                "reasoning": f"The move to {pos} is illegal because this position is already occupied by a {occupied_by} piece. A move can only be made to an empty square."
            }
        }

    ret_game['cot_analysis'] = analysis
    return ret_game, data_cnt


while data_cnt < OCCUPIED_NUM:
    ret_game, game_data_cnt = sample_steps_occupied(data)
    negative_data.append(ret_game)
    data_cnt += game_data_cnt

def sample_steps_no_flank(record):
    game = random.choice(record)
    analysis = game['cot_analysis'][:40]
    ret_game = deepcopy(game)
    del ret_game['cot_analysis']

    data_cnt = 0
    for step in analysis:
        if step['cot_format'] != 'valid_json':
            continue
        data_cnt += 1
        board_state_split = step['board_state'].replace(' ', '').split('\n')
        while True:
            row, col = random.randint(0, 7), random.randint(0, 7)
            if board_state_split[row][col] != '.':
                continue
            pos = coord_to_position(row, col)
            if check_captures_from_string(step['board_state'], pos, step['player'])['valid_move'] == True:
                continue
            break
            
        player = 'Black' if step['player'] == 1 else 'White'
        opponent = 'White' if step['player'] == 1 else 'Black'

        step['position'] = pos
        step['cot'] = {
            "analysis": {
                "chosen_move": pos,
                "is_legal": False,
                "error_type": "No Opponent Pieces Flipped",
                "reasoning": f"The move to {pos} is illegal. Although the square is empty, placing a {player} piece here does not flank or capture any of the opponent's {opponent} pieces."
            }
        }   

    ret_game['cot_analysis'] = analysis
    return ret_game, data_cnt

while data_cnt < OCCUPIED_NUM + NO_FLANK_NUM:
    ret_game, game_data_cnt = sample_steps_no_flank(data)
    negative_data.append(ret_game)
    data_cnt += game_data_cnt

with open('./data/othello_with_cot_negative_only.json','w') as f:
    json.dump(negative_data, f, indent=2, ensure_ascii=False)

