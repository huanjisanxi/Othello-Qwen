import json
from tqdm import tqdm
import argparse
import random

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.env.othello_game import Othello
from src.utils.data_loader import load_csv
from src.data_process.cot_core import generate_rule_based_cot, generate_strategic_cot_task3
from src.utils.api_client import OpenAIClient

def create_training_data(args):
    """
    [V3] Main orchestrator for generating training data.
    """
    print(f"Loading raw game data from {args.raw_data_path}...")
    games_data = load_csv(args.raw_data_path)
    games_data = random.sample(games_data, args.max_games)
    random.shuffle(games_data)
    
    tasks_to_run = set(args.tasks)
    
    # 只有在需要生成任务3数据时才初始化API客户端
    api_client = OpenAIClient() if '3' in tasks_to_run else None
    
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for game_data in tqdm(games_data, desc="Processing Games"):
            moves = game_data['moves']
            
            for move_index, ground_truth_move in enumerate(moves):
                game = Othello()
                # Replay game to the state BEFORE the current move
                for i in range(move_index):
                    try:
                        game.move(moves[i])
                    except ValueError:
                        print(f"Skipping invalid move sequence in game {game_data['id']}.")
                        continue
                
                # --- Generate Task 1 & 2 Data (Rule-based) ---
                if '1' in tasks_to_run or '2' in tasks_to_run or '3' in tasks_to_run:
                    # try:
                    rule_based_cot = generate_rule_based_cot(game)
                    task1_cot = rule_based_cot['task1_cot']
                    task2_cot = rule_based_cot['task2_cot']
                    # except Exception as e:
                    #     print(f"Skipping step in game {game_data['id']} due to rule-based generation error: {e}")
                    #     continue

                # --- Write Task 1 Data ---
                if '1' in tasks_to_run:
                    prompt1_content = f"Task: Analyze Sampled Squares and Identify Plausible Candidates\nPlayer to move: {game.current_player.capitalize()}\nOpponent: {game.current_opponent}\nBoard State:\n{{\n  \"black_pieces\": {sorted(list(game.black))},\n  \"white_pieces\": {sorted(list(game.white))}\n}}\n\nAnalyze a diverse sample of squares to determine which are plausible candidates for a legal move. A plausible candidate must be an empty square adjacent to an opponent's piece. Conclude with a final_plausible_candidates list containing only the squares identified as plausible."
                    f_out.write(json.dumps({"prompt": prompt1_content, "completion": json.dumps(task1_cot, indent=2)}) + '\n')
                
                # --- Write Task 2 Data ---
                if '2' in tasks_to_run:
                    prompt2_content = f"Task: Analyze Plausible Candidates for Legality\nPlayer to move: {game.current_player.capitalize()}\nOpponent: {game.current_opponent}\nBoard State:\n{{\n  \"black_pieces\": {sorted(list(game.black))},\n  \"white_pieces\": {sorted(list(game.white))}\n}}\nPlausible Candidates to Analyze:\n{task1_cot['final_plausible_candidates']}\n\nFor each plausible candidate, determine if it is a legal move by checking the flanking rule. Your analysis must cover every candidate. Conclude with a `final_legal_moves` list containing only the moves confirmed as legal."
                    f_out.write(json.dumps({"prompt": prompt2_content, "completion": json.dumps(task2_cot, indent=2)}) + '\n')

                # --- Generate and Write Task 3 Data (API-based) ---
                if '3' in tasks_to_run:
                    legal_moves = task2_cot['final_legal_moves']
                    if ground_truth_move not in legal_moves:
                        print(f"Warning: Ground truth move {ground_truth_move} not in generated legal moves for game {game_data['id']}. Skipping Task 3.")
                        continue
                    
                    task3_cot = generate_strategic_cot_task3(game, legal_moves, ground_truth_move, api_client)
                    if task3_cot: # If API call was successful
                        prompt3_content = f"Task: Select the Best Strategic Move\nPlayer to move: {game.current_player.capitalize()}\nBoard State:\n{{\n  \"black_pieces\": {sorted(list(game.black))},\n  \"white_pieces\": {sorted(list(game.white))}\n}}\nLegal Moves:\n{legal_moves}\n\nFrom the list of legal moves, determine which move is the absolute best and provide a step-by-step reasoning for your choice, explaining why it is superior to some other alternatives."
                        prompt_task3 = [{"role": "user", "content": prompt3_content}]
                        completion_task3 = [{"role": "assistant", "content": json.dumps(task3_cot, indent=2)}]
                        f_out.write(json.dumps({"prompt": prompt_task3, "completion": completion_task3}) + '\n')

    print(f"Training data generation complete. Output at {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate CoT training data for Othello.")
    parser.add_argument('--raw_data_path', type=str, default='data/othello_dataset.csv', help='Path to the raw CSV game data.')
    parser.add_argument('--output_path', type=str, default='data/test_data_tasks_1_2.jsonl', help='Path to save the generated JSONL file.')
    parser.add_argument('--max_games', type=int, default=10, help='Maximum number of games to process from the CSV.')
    parser.add_argument('--tasks', type=str, default='1,2', help='Comma-separated list of tasks to generate data for (e.g., "1,2", "3", "1,2,3").')
    
    args = parser.parse_args()
    create_training_data(args)
