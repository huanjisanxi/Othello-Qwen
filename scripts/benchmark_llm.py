import argparse
import json
from tqdm import tqdm
import random

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.env.othello_game import Othello
from src.utils.data_loader import load_csv
from src.utils.api_client import OpenAIClient


def run_llm_benchmark(api_client: OpenAIClient, test_games: list):
    """
    评估一个大模型在识别合法走法任务上的 Zero-Shot 表现。
    """
    # (评估指标的定义和 evaluation_loop 与 evaluate_agent.py 中的相同)
    total_correct = 0
    total_wrong = 0
    total_missed = 0
    positions_processed = 0

    print(f"Starting LLM benchmark ...")

    for game_data in tqdm(test_games, desc="Benchmarking Games"):
        
        moves = game_data['moves']
        if not moves: continue
        move_index = random.randint(0, len(moves) - 1)
        
        game = Othello()
        try:
            for i in range(move_index):
                game.move(moves[i])
        except ValueError:
            continue

        true_legal_moves = set(game.get_valid_moves())
        
        player_str = game.current_player.capitalize()
        opponent_str = "white" if player_str == "black" else "black"
        board_json_str = json.dumps({"black_pieces": sorted(list(game.black)), "white_pieces": sorted(list(game.white))})

        # --- Step 1: Identify Plausible Candidates (模拟微调流程) ---
        prompt1 = f"""Task: Analyze Sampled Squares and Identify Plausible Candidates
                Player to move: {player_str}
                Opponent: {opponent_str}
                Board State:
                {board_json_str}

                Analyze a diverse sample of squares to determine which are plausible candidates for a legal move. A plausible candidate must be an empty square adjacent to an opponent's piece. Conclude with a final_plausible_candidates list containing only the squares identified as plausible.
                After analyze Please output the following JSON structure:
                {{
                    "final_plausible_candidates" = ["a1", "b2", "c3", "d4"],
                }}
                """
        
        for i in range(3):
            try:
                response1_text = api_client.generate_response(prompt1)
                json_str1 = response1_text[response1_text.find('{'):response1_text.rfind('}')+1]
                task1_output = json.loads(json_str1)
                plausible_candidates = task1_output.get("final_plausible_candidates", [])
                break
            except Exception as e:
                pass
        else:
            print(f"LLM failed Task 1 for game {game_data['id']}.")
            continue

        prompt2 = f"""Task: Analyze Plausible Candidates for Legality
                Player to move: {player_str}
                Opponent: {opponent_str}
                Board State:
                {board_json_str}
                Plausible Candidates to Analyze:
                {plausible_candidates}
                For each plausible candidate, determine if it is a legal move by checking the flanking rule. Your analysis must cover every candidate. Conclude with a `final_legal_moves` list containing only the moves confirmed as legal.
                After analyze Please output the following JSON structure:
                {{
                    "final_legal_moves":["a1", "b2", "c3", "d4"],
                }}
                """
        
        for i in range(3):
            try:
                response2_text = api_client.generate_response(prompt2)
                json_str2 = response2_text[response2_text.find('{'):response2_text.rfind('}')+1]
                task2_output = json.loads(json_str2)
                predicted_legal_moves = set(task2_output.get("final_legal_moves", []))
                break
            except Exception as e:
                pass
        else:
            print(f"LLM failed Task 2 for game {game_data['id']}.")
            continue


        # (对比和统计逻辑与 evaluate_agent.py 相同)
        correct_predictions = true_legal_moves.intersection(predicted_legal_moves)
        wrong_predictions = predicted_legal_moves - true_legal_moves
        missed_predictions = true_legal_moves - predicted_legal_moves
        total_correct += len(correct_predictions)
        total_wrong += len(wrong_predictions)
        total_missed += len(missed_predictions)
        positions_processed += 1
    
    # (打印结果的逻辑与 evaluate_agent.py 相同)
    print("\n--- LLM Zero-Shot Benchmark Results ---")
    precision = total_correct / (total_correct + total_wrong) if (total_correct + total_wrong) > 0 else 0
    recall = total_correct / (total_correct + total_missed) if (total_correct + total_missed) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Positions evaluated: {positions_processed}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
    print("---------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark a large language model on Othello legal move identification.")
    # 您可以添加参数来选择不同的教师模型
    parser.add_argument('--test_data_path', type=str, default='data/othello_dataset.csv', help='Path to the test game data.')
    parser.add_argument('--num_positions', type=int, default=500, help='Number of random positions to evaluate.')
    
    args = parser.parse_args()
    
    api_client = OpenAIClient()
    test_games = load_csv(args.test_data_path)
    test_games = random.sample(test_games, args.num_positions)
    run_llm_benchmark(api_client, test_games)