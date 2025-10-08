import csv
import re
from ..env.othello_game import Othello
from datasets import Dataset, DatasetDict
import json
from tqdm import tqdm

def load_and_prepare_dataset(jsonl_path, split_ratio=0.9):
    """
    加载JSONL文件并转换为Hugging Face Dataset对象
    
    Args:
        jsonl_path: JSONL文件路径
        split_ratio: 训练集与验证集的划分比例
        
    Returns:
        DatasetDict: 包含训练集和验证集的DatasetDict对象
    """
    # 读取JSONL文件
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="load data"):
            try:
                item = json.loads(line)
                # 提取对话内容
                user_message = item["prompt"][0]["content"]
                assistant_message = item["completion"][0]["content"]
                
                # 转换为训练所需的格式
                data.append({
                    "prompt": user_message,
                    "completion": assistant_message,
                })
            except Exception as e:
                print(f"error: {e}")
                continue
    
    # 转换为Dataset对象
    dataset = Dataset.from_list(data)
    
    # 划分训练集和验证集
    dataset_split = dataset.train_test_split(test_size=1-split_ratio, shuffle=True)
    
    return dataset_split

def parse_moves(move_str):
    """Parse move string like "f5d6c4" into list ["f5", "d6", "c4"]"""
    return re.findall('.{2}', move_str)

def play_moves(moves, show_steps=True):
    """Simulate game from move list and print process"""
    game = Othello()
    print("Initial board:")
    game.print()
    print(f"Black's initial valid moves: {game.get_valid_moves()}\n")
    
    try:
        for i, coord in enumerate(moves, 1):
            print(f"Move {i}: {game.current_player} plays {coord}")
            flipped = game.move(coord)
            print(f"Flipped stones: {flipped}")
            if show_steps:
                game.print()
                
        print("Game over!")
        print("Final board:")
        game.print()
        
        winner = game.get_winner()
        print(f"Winner: {winner}" if winner else "It's a tie")
        print(f"Stone counts - Black: {len(game.black)}, White: {len(game.white)}")
        
    except ValueError as e:
        print(f"Error: {e}")
    
    return game

def load_csv(filename, max_games=None):
    """Load game data from CSV file"""
    games = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_games and i >= max_games:
                break
            games.append({
                'id': row['eOthello_game_id'],
                'winner': 'black' if row['winner'] == '1' else 'white',
                'moves': parse_moves(row['game_moves'])
            })
    return games
