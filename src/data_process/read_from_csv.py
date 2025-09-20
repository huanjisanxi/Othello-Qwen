import csv
import json
import argparse
import os

def position_to_coord(pos):
    col, row = pos[0], pos[1]
    row = int(row) - 1  
    col = ord(col) - ord('a') 
    return row, col

def inside(r, c):
    return 0 <= r < 8 and 0 <= c < 8

def apply_move(board, pos, player):
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

def split_moves(move_string):
    moves = [move_string[i:i+2] for i in range(0, len(move_string), 2)]
    return moves

def get_move_players(moves):
    board = [[0] * 8 for _ in range(8)]
    board[3][3], board[3][4] = 2, 1 
    board[4][3], board[4][4] = 1, 2
    
    current_player = 1  
    move_players = []
    
    for pos in moves:
        if apply_move([row.copy() for row in board], pos, current_player):
            move_players.append(current_player)
            apply_move(board, pos, current_player)
            current_player = 3 - current_player 
        else:
            current_player = 3 - current_player
            move_players.append(current_player)
            apply_move(board, pos, current_player)
            current_player = 3 - current_player  
    
    return move_players

def read_othello_data(file_path):
    othello_data = []
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            moves = split_moves(row['game_moves'])
            move_players = get_move_players(moves)
            
            game_data = {
                'eOthello_game_id': int(row['eOthello_game_id']),
                'winner': int(row['winner']),
                # 'game_moves': row['game_moves'],
                # 'parsed_moves': moves,
                'move_count': len(moves),
                # 'move_players': move_players,  
                'moves_with_players': list(zip(moves, move_players)) 
            }
            othello_data.append(game_data)
    
    return othello_data
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    othello_games = read_othello_data(args.input)
    
    if othello_games:
        output_filename = os.path.join(args.output, 'othello_with_players.json')
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(othello_games, f, ensure_ascii=False, indent=2)
        print(f"\nsave at {output_filename}")