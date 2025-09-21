from typing import List, Dict, Any, Optional

def position_to_coord(pos: str) -> tuple:
    col, row = pos[0], pos[1]
    row_idx = int(row) - 1
    col_idx = ord(col) - ord('a') 
    return row_idx, col_idx

def coord_to_position(r: int, c: int) -> str:
    col = chr(c + ord('a'))
    row = str(r + 1)
    return col + row

def inside(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8

def board_string_to_list(board_str: str) -> List[List[int]]:
    board = []
    for line in board_str.split('\n'):
        row = []
        for cell in line.strip().split():
            if cell == '.':
                row.append(0)
            elif cell == 'B':
                row.append(1)
            elif cell == 'W':
                row.append(2)
        board.append(row)
    return board

def apply_move(board: List[List[int]], pos: str, player: int) -> bool:
    r, c = position_to_coord(pos)
    if board[r][c] != 0:
        return False 
    
    opponent = 3 - player  
    flipped = []
    
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    
    valid = False
    for dr, dc in directions:
        path = []
        nr, nc = r + dr, c + dc
        while inside(nr, nc) and board[nr][nc] == opponent:
            path.append((nr, nc))
            nr += dr
            nc += dc
        if path and inside(nr, nc) and board[nr][nc] == player:
            valid = True
            flipped.extend(path)
    
    if not valid:
        return False
    
    board[r][c] = player
    for fr, fc in flipped:
        board[fr][fc] = player
    return True

def check_captures_from_string(board_str: str, pos: str, player: int) -> Dict[str, Any]:
    board = board_string_to_list(board_str)
    return check_captures(board, pos, player)

def check_captures(board: List[List[int]], pos: str, player: int) -> Dict[str, Any]:
    r, c = position_to_coord(pos)
    
    # 检查位置是否有效且为空
    if not inside(r, c) or board[r][c] != 0:
        return {
            "position": pos,
            "player": player,
            "valid_move": False,
            "capture_situations": []
        }
    
    opponent = 3 - player
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    
    capture_info = {
        "position": pos,
        "player": player,
        "valid_move": False,
        "capture_situations": []
    }
    
    for dr, dc in directions:
        captured = [] 
        nr, nc = r + dr, c + dc
        
        if inside(nr, nc) and board[nr][nc] == opponent:
            captured.append((nr, nc))
            nr += dr
            nc += dc
            
            while inside(nr, nc):
                if board[nr][nc] == 0:  
                    captured = []
                    break
                if board[nr][nc] == player: 
                    if captured:
                        friendly_pos = coord_to_position(nr, nc)
                        captured_positions = [coord_to_position(r_pos, c_pos) 
                                            for r_pos, c_pos in captured]
                        
                        capture_info["capture_situations"].append({
                            "direction": (dr, dc),
                            "friendly_piece": friendly_pos,
                            "captured_pieces": captured_positions,
                            "count": len(captured_positions)
                        })
                        capture_info["valid_move"] = True
                    break
                captured.append((nr, nc))
                nr += dr
                nc += dc
    
    return capture_info

def print_board(board: List[List[int]]) -> None:
    symbols = {0: ".", 1: "B", 2: "W"}
    for row in board:
        print(" ".join(symbols[cell] for cell in row))
    print()

def convert_capture_to_english_cot(capture_data: Dict) -> Dict[str, str]:

    player_name = "Black" if capture_data["player"] == 1 else "White"
    player_symbol = "B" if capture_data["player"] == 1 else "W"
    opponent_name = "White" if capture_data["player"] == 1 else "Black"
    opponent_symbol = "W" if capture_data["player"] == 1 else "B"
    
    reasoning: List[str] = []
    reasoning.append(f"Analyzing the move: {player_name} ({player_symbol}) at {capture_data['position']}.")
    
    if capture_data["valid_move"]:
        reasoning.append(f"This is a valid move because it can capture {opponent_name} pieces through flanking.")
        
        for i, situation in enumerate(capture_data["capture_situations"], 1):
            direction = _get_english_direction(situation["direction"])
            captured = ", ".join(situation["captured_pieces"])
            
            reasoning.append(
                f"Situation {i}: Along the {direction}, the new {player_symbol} at {capture_data['position']} "
                f"flanks with an existing {player_symbol} at {situation['friendly_piece']}, "
                f"capturing {opponent_symbol} pieces at {captured} (total {situation['count']})."
            )
        
        total_captured = sum(s["count"] for s in capture_data["capture_situations"])
        reasoning.append(
            f"Overall, this move is legal and will capture {total_captured} {opponent_symbol} pieces, "
        )
    else:
        reasoning.append(
            f"This is an invalid move because it cannot capture any {opponent_name} pieces. "
            f"Othello rules require a move to flip at least one opponent's piece to be valid."
        )
    
    return " ".join(reasoning)

def _get_english_direction(direction: tuple) -> str:
    direction_map = {
        (-1, -1): "northwest diagonal",
        (-1, 0): "north (vertical)",
        (-1, 1): "northeast diagonal",
        (0, -1): "west (horizontal)",
        (0, 1): "east (horizontal)",
        (1, -1): "southwest diagonal",
        (1, 0): "south (vertical)",
        (1, 1): "southeast diagonal"
    }
    return direction_map.get(direction, f"direction {direction}")


if __name__ == '__main__':
    input_data = {
        "position": "b5",
        "player": 1,
        "board_state": ". . . . . . . .\n. . . . . . . .\n. . . W W W . .\n. W W W B W . .\n. . W W B W . .\n. . B W W W . .\n. . . W B . . .\n. . . . . . . ."
    }
    
    result = check_captures_from_string(
        input_data["board_state"],
        input_data["position"],
        input_data["player"]
    )
    
    print(json.dumps(result, indent=2))
    