import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Optional

from src.env.othello_game import Othello 


class OthelloAgent:
    def __init__(self, base_model_id: str, adapter_path: str, device: str = "auto"):
        print("Initializing Othello Agent...")
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading base model: {base_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16, 
            device_map=self.device,
            trust_remote_code=True
        )
        
        print(f"Loading LoRA adapter from: {adapter_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.eval() 
        print(f"Agent initialized on device: {self.device}")

    
    def _create_prompt(self, task_name: str, game: Othello, **kwargs) -> str:
        player_str = game.current_player.capitalize()
        opponent_str = "black" if player_str == "white" else "white"
        board_json_str = json.dumps({
            "black_pieces": sorted(list(game.black)),
            "white_pieces": sorted(list(game.white))
        }, indent=2)

        if task_name == "Task1":
            return f"""Task: Analyze Sampled Squares and Identify Plausible Candidates
                Player to move: {player_str}
                Opponent: {opponent_str}
                Board State:
                {board_json_str}

                Analyze a diverse sample of squares to determine which are plausible candidates for a legal move. A plausible candidate must be an empty square adjacent to an opponent's piece. Conclude with a final_plausible_candidates list containing only the squares identified as plausible."""
        
        elif task_name == "Task2":
            plausible_candidates = kwargs.get("plausible_candidates", [])
            return f"""Task: Analyze Plausible Candidates for Legality
                Player to move: {player_str}
                Opponent: {opponent_str}
                Board State:
                {board_json_str}
                Plausible Candidates to Analyze:
                {plausible_candidates}
                For each plausible candidate, determine if it is a legal move by checking the flanking rule. Your analysis must cover every candidate. Conclude with a `final_legal_moves` list containing only the moves confirmed as legal."""
        
        else:
            raise ValueError(f"Unknown task name: {task_name}")

    def analyze_position(self, game: Othello) -> Dict:
        analysis_result = {
            "plausible_candidates": None,
            "predicted_legal_moves_analysis": {},
            "predicted_legal_moves": [],
            "chosen_move": None,
            "errors": []
        }

        with torch.no_grad():
            # --- Step 1: Identify Plausible Candidates ---
            try:
                # (The code for Step 1 is the same as before)
                prompt1 = self._create_prompt("Task1", game)
                inputs1 = self.tokenizer(prompt1, return_tensors="pt").to(self.device)
                outputs1 = self.model.generate(**inputs1, max_new_tokens=512, do_sample=False)
                response1_text = self.tokenizer.decode(outputs1[0], skip_special_tokens=True)
                
                json_str1 = response1_text[response1_text.find('{'):response1_text.rfind('}')+1]
                task1_output = json.loads(json_str1)
                analysis_result["plausible_candidates"] = task1_output.get("final_plausible_candidates", [])
            except Exception as e:
                error_msg = f"Error in Task 1: {e}"
                analysis_result["errors"].append(error_msg)
                print(error_msg)
                return analysis_result

            # --- Step 2: Filter for Legal Moves ---
            try:
                # (The code for Step 2 is the same as before)
                prompt2 = self._create_prompt("Task2", game, plausible_candidates=analysis_result["plausible_candidates"])
                inputs2 = self.tokenizer(prompt2, return_tensors="pt").to(self.device)
                outputs2 = self.model.generate(**inputs2, max_new_tokens=1024, do_sample=False)
                response2_text = self.tokenizer.decode(outputs2[0], skip_special_tokens=True)

                json_str2 = response2_text[response2_text.find('{'):response2_text.rfind('}')+1]
                task2_output = json.loads(json_str2)
                
                detailed_analysis = task2_output.get("detailed_analysis", [])
                legal_moves_list = []
                for move_analysis in detailed_analysis:
                    if move_analysis.get("is_legal"):
                        pos = move_analysis.get("position")
                        flipped_count = len(move_analysis.get("flipped_stones", []))
                        analysis_result["predicted_legal_moves_analysis"][pos] = flipped_count
                        legal_moves_list.append(pos)
                analysis_result["predicted_legal_moves"] = sorted(legal_moves_list)

            except Exception as e:
                error_msg = f"Error in Task 2: {e}"
                analysis_result["errors"].append(error_msg)
                print(error_msg)
                return analysis_result

        # --- Step 3 : Greedy Algorithm ---
        if analysis_result["predicted_legal_moves_analysis"]:
            best_move = max(
                analysis_result["predicted_legal_moves_analysis"].items(), 
                key=lambda item: item[1]
            )[0]
            analysis_result["chosen_move"] = best_move
        
        return analysis_result

    # choose_move method remains the same
    def choose_move(self, game: Othello) -> Optional[str]:
        analysis = self.analyze_position(game)
        return analysis.get("chosen_move")
