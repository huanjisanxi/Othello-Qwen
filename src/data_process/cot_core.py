import json
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from utils.api_client import OpenAIClient
from utils.othello_utils import position_to_coord, inside, apply_move, print_board

load_dotenv()  

class CoTGenerator:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "deepseek-v3"):
        self.client = OpenAIClient(api_key=api_key, base_url=base_url, model=model)
        
    def simulate_board(self, moves_with_players: List[tuple]) -> List[List[int]]:
        board = [[0]*8 for _ in range(8)]
        board[3][3], board[3][4] = 2, 1 
        board[4][3], board[4][4] = 1, 2 
        for move, player in moves_with_players:
            apply_move(board, move, player) 
        return board
    
    def generate_cot_for_move(self, game_data: Dict[str, Any], move_index: int) -> Dict[str, Any]:
        moves_with_players = game_data.get("moves_with_players", [])
        if move_index >= len(moves_with_players):
            return {
                "move_index": move_index,
                "position": None,
                "player": None,
                "board_state": None,
                "cot_english": "Invalid move index"
            }
        
        current_move, current_player = moves_with_players[move_index]
        player_str = "Black" if current_player == 1 else "White"
        move_number = move_index + 1
        
        previous_moves = moves_with_players[:move_index]
        board = self.simulate_board(previous_moves)
        
        def board_to_string(b):
            symbols = {0: ".", 1: "B", 2: "W"}
            return "\n".join([" ".join(symbols[cell] for cell in row) for row in b])
        
        board_str = board_to_string(board)
        
        recent_moves = previous_moves[-5:]  
        moves_summary = "\n".join([
            f"Move {i+1 + (move_index - len(recent_moves))}: {move} by {'Black' if p == 1 else 'White'}"
            for i, (move, p) in enumerate(recent_moves)
        ])
        
        prompt = f"""
        # Role
        You are a world-class Othello grandmaster with a steady playing style. You prioritize long-term positioning and restricting the opponent over greedily capturing a large number of discs in the short term.

        # Task
        Your task is to provide a professional, structured thought process analysis for the specified move in the following Othello game.

        ---

        # Game Information
        - **Current Player (You)**: {player_str}
        - **Move Position**: {current_move}
        - **Move Number**: {move_number}
        - **Current Board State** (B=Black, W=White, .=Empty):
        {board_str}
        - **Recent Moves**:
        {moves_summary if moves_summary else "None"}

        ---

        # Task Requirements
        Please thoroughly analyze why `{current_move}` is a wise choice, and provide your thought process in strict accordance with the JSON format defined below. **Do not output any additional explanations, comments, or opening remarks—only the JSON object itself**.

        # Output Format (JSON)
        ```json
        {{
        "analysis": {{
            "board_assessment": {{
            "strategic_overview": "A macro assessment of the current situation, e.g., which player has the advantage, whether the game is in the opening/midgame/endgame, and the main strategic focus.",
            "key_positions": "Identify the control status of key positions on the board, especially the stability of the four **corners** and **edges**.",
            "mobility": "Analyze and compare the **mobility** of both players (i.e., the number of legal moves available) and assess which player will take the initiative in the next few moves."
            }},
            "move_justification": {{
            "position": "{current_move}",
            "flipped_discs_count": "The number of discs flipped by this move.",
            "strategic_value": [
                "Explain the core strategic value of this move (this is an array, which can contain multiple items). E.g.: Whether it occupies favorable terrain (such as central positions), builds a solid edge, gains potential control over opposite corners, or effectively **restricts the opponent's mobility**."
            ]
            }},
            "alternatives_considered": [
            {{
                "position": "Another position you considered, e.g., 'C3'",
                "reason_for_rejection": "Briefly explain why you ultimately abandoned this position. E.g.: 'Although it captures more discs, it would prematurely give up a corner, allowing the opponent to gain a large number of stable discs' or 'This position would severely reduce our mobility'."
            }},
            ],
            "opponent_response_prediction": {{
            "most_likely_moves": [
                {{
                "position": "The position you predict the opponent is most likely to respond with",
                "reasoning": "Explain why you think the opponent will make this move, e.g.: 'This is the opponent's only available move' or 'This move can maximize the recapture of edge control'."
                }}
            ],
            "long_term_outlook": "A brief outlook on the future direction of the game after the opponent responds."
            }}
        }}
        }}
        """
        
        try:
            cot = self.client.generate_response(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2048
            )
            cleaned_cot = re.sub(r'^\s*```json\s*', '', cot, flags=re.MULTILINE)
            cleaned_cot = re.sub(r'\s*```\s*$', '', cleaned_cot, flags=re.MULTILINE)
            cleaned_cot = re.sub(r'\n+', '\n', cleaned_cot).strip()
            json.loads(cleaned_cot)  
            return {
                "move_index": move_index,
                "position": current_move,
                "player": current_player,
                "board_state": board_str,
                "cot_english": cleaned_cot, 
                "cot_format": "valid_json"
            }
        except json.JSONDecodeError as e:
            print(f"CoT output is not valid JSON for move {move_number}: {e}")
            return {
                "move_index": move_index,
                "position": current_move,
                "player": current_player,
                "board_state": board_str,
                "cot_english": cot,
                "cot_format": "invalid_json"
            }
        except Exception as e:
            print(f"Error generating CoT for move {move_number}: {e}")
            return {
                "move_index": move_index,
                "position": current_move,
                "player": current_player,
                "board_state": board_str,
                "cot_english": f"Error: {str(e)}",
                "cot_format": "error"
            }

def process_single_game(game_data: Dict[str, Any], max_workers: int = 5) -> Dict[str, Any]:
    moves_with_players = game_data.get("moves_with_players", [])
    moves_count = len(moves_with_players)
    cot_generator = CoTGenerator()
    cot_list = []
    for move_idx in range(moves_count):
        cot_result = cot_generator.generate_cot_for_move(game_data, move_idx)
        cot_list.append(cot_result)
    cot_list.sort(key=lambda x: x.get("move_index", 0))
    game_data["cot_analysis"] = cot_list
    return game_data