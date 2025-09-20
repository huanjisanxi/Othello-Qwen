from typing import List 

def position_to_coord(pos: str) -> tuple:
    col, row = pos[0], pos[1]
    row_idx = int(row) - 1
    col_idx = ord(col) - ord('a') 
    return row_idx, col_idx

def inside(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8

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

def print_board(board: List[List[int]]) -> None:
    symbols = {0: ".", 1: "B", 2: "W"}
    for row in board:
        print(" ".join(symbols[cell] for cell in row))
    print()
