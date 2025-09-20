# scripts/cot_generate.py
import json
import os
import sys
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_process.cot_core import CoTGenerator, process_single_game 

def process_game_data(
    file_path: str, 
    output_path: str, 
    max_games: Optional[int] = None, 
    max_workers: int = 5
) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        game_data_list = json.load(f)
    
    if max_games:
        game_data_list = game_data_list[:max_games]
    
    total_games = len(game_data_list)
    cot_generator = CoTGenerator()  
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_game, game_data, max_workers)
            for game_data in game_data_list
        ]
        
        for i, future in enumerate(tqdm(futures, total=total_games, desc="Processing games")):
            processed_game = future.result()
            game_data_list[i] = processed_game 
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(game_data_list, f, ensure_ascii=False, indent=2)
    print(f"Completed! Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CoT with board state for Othello moves")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--max-games", type=int, help="Max games to process")
    parser.add_argument("--max-workers", type=int, default=5, help="Number of parallel workers")
    
    args = parser.parse_args()
    process_game_data(args.input, args.output, args.max_games, args.max_workers)