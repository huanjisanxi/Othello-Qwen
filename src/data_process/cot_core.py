import random
import json

import flash_attention
from numpy import flip
from peft import TaskType
from src.env.othello_game import Othello
from src.utils.api_client import OpenAIClient

def _find_flank_details(game: Othello, pos: str) -> dict:
    """
    一个辅助函数，用于找到形成夹击的具体己方和对方棋子。
    这是对 game._get_flips 的增强，以提供更丰富的推理信息。
    """
    row, col = game._parse_coord(pos)
    if row is None or pos in game.black or pos in game.white:
        return {}

    current, opponent = (game.black, game.white) if game.current_player == 'black' else (game.white, game.black)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    flank_details = {}
    
    for dr, dc in directions:
        line = []
        r, c = row + dr, col + dc
        adja_pos = game._to_coord(r, c) if game._to_coord(r, c) in opponent else None
        
        while 0 <= r < game.size and 0 <= c < game.size:
            current_pos = game._to_coord(r, c)
            if current_pos in opponent:
                line.append(current_pos)
            elif current_pos in current:
                if line: # Found an anchor piece
                    flank_details[adja_pos] = (line, current_pos) 
                break
            else: # Empty square
                if adja_pos is not None:
                    flank_details[adja_pos] = ([], current_pos)    
                break
            r, c = r + dr, c + dc
        else:
            if adja_pos is not None:
                flank_details[adja_pos] = ([], current_pos) 
            
    return flank_details

def generate_rule_based_cot(game: Othello) -> dict:
    """
    [V3] 为任务一和二生成带有超详细分析的、基于规则的 CoT 数据。
    """
    opponent_pieces = game.white if game.current_player == 'black' else game.black
    all_squares = {game._to_coord(r, c) for r in range(game.size) for c in range(game.size)}
    occupied_squares = game.black | game.white
    
    # --- 任务一：识别和分析候选点 ---
    plausible_candidates = set()
    adjacencies = {pos: [] for pos in all_squares}

    for r_idx in range(game.size):
        for c_idx in range(game.size):
            pos = game._to_coord(r_idx, c_idx)
            if pos in occupied_squares: continue
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < game.size and 0 <= nc < game.size:
                        adj_pos = game._to_coord(nr, nc)
                        if adj_pos in opponent_pieces:
                            plausible_candidates.add(pos)
                            adjacencies[pos].append(adj_pos)

    # 条件性负采样
    analysis_points = set(plausible_candidates)
    while len(analysis_points) <= 10:
        if len(all_squares-analysis_points-occupied_squares) != 0:
            selected_group = random.choice([list(all_squares-analysis_points-occupied_squares), list(occupied_squares)])
        else:
            selected_group = list(occupied_squares)
        analysis_points.add(random.choice(selected_group))

    task1_analysis = {}
    # something wrong???
    # should be for pos in sort(list(analysis_points)) ???
    analysis_points = list(analysis_points)
    random.shuffle(analysis_points)
    for pos in analysis_points:
        if pos in occupied_squares:
            reason = f"Illegal: Position is already occupied by a {'black' if pos in game.black else 'white'} piece."
        elif pos not in plausible_candidates:
            reason = "Invalid Candidate: Position is empty but not adjacent to any opponent pieces."
        else:
            reason = f"Plausible Candidate: Position is empty and adjacent to opponent piece(s) at {', '.join(sorted(adjacencies[pos]))}. Legality needs to be checked."
    
        task1_analysis[pos] = reason

    plausible_candidates = sorted(list(plausible_candidates))
    task1_cot = {
        "analysis": task1_analysis,
        "final_plausible_candidates": plausible_candidates
    }

    # --- 任务二：在合理的候选点中分析出合法落子 ---
    if len(plausible_candidates) > 10:
        plausible_candidates = plausible_candidates[:10]
    task2_analysis_details = {}
    final_legal_moves = {}
    for pos in plausible_candidates:
        flank_details = _find_flank_details(game, pos)
        flipped_pieces = sum((v[0] for v in flank_details.values()), [])
        reason = f"Adjacent to opponent piece(s) at {', '.join(sorted(adjacencies[pos]))}."
        for adaj_pos, flipped_and_anchor in flank_details.items():
            flipped, anchor = flipped_and_anchor
            if len(flipped) == 0:
                reason += f"in the direction of {adaj_pos}, flanks no {game.current_opponent} pieces, "
            else:
                reason += f"in the direction of {adaj_pos}, flanks {', '.join(sorted(flipped))} ({game.current_opponent} pieces) with anchor piece at {anchor} ({game.current_player} pieces), "
        conclusion = f"Position is invalid, flanks no pieces" if len(flipped_pieces) == 0 else \
                f"Position is valid, flanks {len(flipped_pieces)} {game.current_opponent} pieces: {flipped_pieces}"
        reason += conclusion
        task2_analysis_details[pos] = reason
        if len(flipped_pieces) > 0:
            final_legal_moves[pos] = flipped_pieces

    ground_truth_legal_moves = game.get_valid_moves()
    assert set(final_legal_moves).issubset(set(ground_truth_legal_moves)), \
        f"Mismatch! Generated: {sorted(final_legal_moves)}, Ground Truth: {sorted(ground_truth_legal_moves)}"

    task2_cot = {
        "detailed_analysis": task2_analysis_details,
        "final_legal_moves": final_legal_moves
    }

    return {"task1_cot": task1_cot, "task2_cot": task2_cot}


def generate_strategic_cot_task3(game: Othello, legal_moves: list, ground_truth_move: str, api_client: OpenAIClient) -> dict:
    prompt = f"""You are a world-class Othello grandmaster. Your task is to analyze the board state and a list of legal moves, then explain why the given expert's choice is strategically superior.

# Context
- Player to move: {game.current_player.capitalize()}
- Board State:
  - Black Pieces: {sorted(list(game.black))}
  - White Pieces: {sorted(list(game.white))}
- All Legal Moves: {legal_moves}
- The Expert's Choice: {ground_truth_move}

# Task
Provide a structured analysis in JSON format explaining why the expert's choice is the best move among all legal options. Focus on long-term strategic concepts like corner acquisition, edge stability, mobility restriction, and parity. Do not just focus on the number of flipped discs.

# JSON Output Format
{{
  "strategic_analysis": {{
    "best_move": "{ground_truth_move}",
    "core_reasoning": "A concise, high-level explanation of the move's primary strategic advantage.",
    "comparison_with_alternatives": [
      {{
        "alternative_move": "An alternative legal move.",
        "why_inferior": "Explain why this alternative is strategically weaker than the expert's choice."
      }}
    ],
    "long_term_goal": "What strategic goal does this move achieve for the next 5-10 turns?"
  }}
}}
"""
    try:
        response_str = api_client.generate_response(prompt, temperature=0.3)
        # 清理和解析JSON
        json_str = response_str[response_str.find('{'):response_str.rfind('}')+1]
        return json.loads(json_str)
    except Exception as e:
        print(f"Failed to generate or parse strategic CoT for move {ground_truth_move}. Error: {e}")
        return None # 返回None表示失败