import csv
import re

class Othello:
    def __init__(self):
        self.size = 8
        self.reset()

    def reset(self):
        """Reset board to initial state with 4 central stones"""
        self.black = {'d5', 'e4'}
        self.white = {'d4', 'e5'}
        self.move_history = []  # Stores comprehensive information of each move
        self.current_player = 'black'  # Black player goes first
        self.game_over = False
        self._record_initial_state()

    def _record_initial_state(self):
        """Record initial board state as step 0"""
        self.move_history.append({
            'step': 0,
            'player': None,  # No player for initial state
            'position': None,  # No move position
            'flipped_stones': [],  # No flipped stones
            'board_state': {
                'black': set(self.black),
                'white': set(self.white)
            },
            'next_player': self.current_player,
            'game_over': self.game_over
        })

    def _parse_coord(self, coord):
        """Convert 'a1' style coordinate to 0-based indices"""
        if len(coord) < 2:
            return None, None
            
        col, row = coord[0].lower(), coord[1:]
        if not col.isalpha() or not row.isdigit():
            return None, None
            
        col_idx = ord(col) - ord('a')
        row_idx = int(row) - 1
        
        return (row_idx, col_idx) if 0 <= col_idx < self.size and 0 <= row_idx < self.size else (None, None)

    def _to_coord(self, row_idx, col_idx):
        """Convert 0-based indices to 'a1' style coordinate"""
        return chr(col_idx + ord('a')) + str(row_idx + 1) if 0 <= row_idx < self.size and 0 <= col_idx < self.size else None

    def _get_flips(self, coord):
        """Calculate which stones will be flipped for a potential move"""
        row, col = self._parse_coord(coord)
        if row is None or col is None or coord in self.black or coord in self.white:
            return []

        current, opponent = (self.black, self.white) if self.current_player == 'black' else (self.white, self.black)
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        
        flips = []
        for dr, dc in directions:
            r, c = row + dr, col + dc
            temp = []
            
            while 0 <= r < self.size and 0 <= c < self.size:
                pos = self._to_coord(r, c)
                
                if pos in opponent:
                    temp.append(pos)
                    r += dr
                    c += dc
                elif pos in current:
                    flips.extend(temp)
                    break
                else:
                    break
        
        return flips

    def get_valid_moves(self):
        """Return list of all valid moves for current player"""
        valid = []
        for row in range(self.size):
            for col in range(self.size):
                coord = self._to_coord(row, col)
                if coord not in self.black and coord not in self.white and self._get_flips(coord):
                    valid.append(coord)
        return valid

    def move(self, coord):
        """
        Place a stone at specified coordinate
        Returns list of flipped stones
        Raises ValueError for invalid moves
        """
        if self.game_over:
            raise ValueError("Game is over")
        
        # Check if position is already occupied
        if coord in self.black or coord in self.white:
            raise ValueError(f"Position {coord} is already occupied")
            
        # Check if coordinate is valid
        row, col = self._parse_coord(coord)
        if row is None or col is None:
            raise ValueError(f"Invalid coordinate: {coord}")
            
        flips = self._get_flips(coord)
        if not flips:
            valid_moves = self.get_valid_moves()
            raise ValueError(f"Invalid move: {coord}, valid moves are: {valid_moves}")

        current_player = self.current_player
        
        # Place stone and flip opponent's stones
        if current_player == 'black':
            self.black.add(coord)
            for pos in flips:
                self.white.remove(pos)
                self.black.add(pos)
        else:
            self.white.add(coord)
            for pos in flips:
                self.black.remove(pos)
                self.white.add(pos)

        next_player = 'white' if current_player == 'black' else 'black'
        
        # Check game over status
        self._check_game_over()
        
        # Update next player if game continues
        if not self.game_over:
            self.current_player = next_player
            if not self.get_valid_moves():
                self.current_player = 'white' if self.current_player == 'black' else 'black'
                if not self.get_valid_moves():
                    self.game_over = True
            next_player = self.current_player

        # Record comprehensive move information
        self.move_history.append({
            'step': len(self.move_history),
            'player': current_player,
            'position': coord,
            'flipped_stones': flips.copy(),
            'board_state': {
                'black': set(self.black),
                'white': set(self.white)
            },
            'next_player': next_player if not self.game_over else None,
            'game_over': self.game_over
        })

        return flips

    def _check_game_over(self):
        """Check if game should end (board full or no valid moves for both players)"""
        if len(self.black) + len(self.white) == self.size * self.size:
            self.game_over = True
        elif not self.get_valid_moves():
            opponent = 'white' if self.current_player == 'black' else 'black'
            self.current_player = opponent
            if not self.get_valid_moves():
                self.game_over = True
            self.current_player = 'white' if opponent == 'black' else 'black'

    def print(self, step=None):
        """Print board state, optionally specify step number (0 for initial state)"""
        if step is not None and 0 <= step < len(self.move_history):
            state = self.move_history[step]['board_state']
            black, white = state['black'], state['white']
        else:
            black, white = self.black, self.white

        print('  ' + ' '.join([chr(ord('a') + i) for i in range(self.size)]))
        for row_idx in range(self.size):
            print(f"{row_idx + 1} ", end='')
            for col_idx in range(self.size):
                coord = self._to_coord(row_idx, col_idx)
                if coord in black:
                    print('B', end=' ')
                elif coord in white:
                    print('W', end=' ')
                else:
                    print('.', end=' ')
            print()
        print()

    def get_winner(self):
        """Return winner ('black' or 'white') or None for tie (only after game over)"""
        if not self.game_over:
            return None
        return 'black' if len(self.black) > len(self.white) else 'white' if len(self.white) > len(self.black) else None

    def get_move_history(self):
        """Return copy of complete move history to prevent external modification"""
        return [step.copy() for step in self.move_history]
    
    def get_current_state(self):
        """Return the latest board state from move_history (last element)"""
        if not self.move_history:
            return None
        # Return a copy to prevent external modification of internal state
        return self.move_history[-1].copy()

    @property
    def current_opponent(self):
        return "white" if self.current_player == "black" else "black"

    def set_board_state(self, board_state, player):
        """
        Set a completely new game state with clean history
        Args:
            board_state: Dictionary with 'black' and 'white' keys, each containing set of coordinates
            player: Current player to set ('black' or 'white')
        """
        # Validate player
        if player not in ['black', 'white']:
            raise ValueError(f"Invalid player: {player}. Must be 'black' or 'white'")
            
        # Validate board_state structure
        if not isinstance(board_state, dict) or 'black' not in board_state or 'white' not in board_state:
            raise ValueError("board_state must contain 'black' and 'white' keys")
            
        # Validate and convert positions to sets
        try:
            black_positions = set(board_state['black'])
            white_positions = set(board_state['white'])
        except:
            raise ValueError("board_state values must be iterable (list, set, etc.)")
            
        # Validate all coordinates
        for coord in black_positions.union(white_positions):
            row, col = self._parse_coord(coord)
            if row is None or col is None:
                raise ValueError(f"Invalid coordinate: {coord}")
                
        # Check for overlapping positions
        overlapping = black_positions.intersection(white_positions)
        if overlapping:
            raise ValueError(f"Overlapping positions: {sorted(overlapping)}")
        
        # 清除所有历史记录，创建全新游戏
        self.move_history = []
        self.game_over = False  # 新游戏状态下游戏未结束
        self.current_player = player
        self.black = black_positions.copy()
        self.white = white_positions.copy()
        
        # 记录新的初始状态（作为第0步）
        self.move_history.append({
            'step': 0,
            'player': None,
            'position': None,
            'flipped_stones': [],
            'board_state': {
                'black': set(self.black),
                'white': set(self.white)
            },
            'next_player': self.current_player,
            'game_over': self.game_over,
            'note': 'New game initialized'  # 标记为新游戏起点
        })


# Utility functions
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

def print_game_from_csv(game_data, show_steps=True):
    """Print a game from CSV data"""
    print(f"\n=== Game ID: {game_data['id']} ===")
    print(f"Reported winner: {game_data['winner']}")
    play_moves(game_data['moves'], show_steps)


# Example usage
if __name__ == "__main__":
    # Corrected move sequence
    # sample_moves = ['f4', 'd6', 'c4', 'e3', 'g5', 'f6']
    # print("=== Move sequence demonstration ===")
    # play_moves(sample_moves)
    
    # To use with CSV file, uncomment:
    print("\n=== CSV file demonstration ===")
    games = load_csv('/data/data_public/zjy/Othello-Qwen/data/othello_dataset.csv', max_games=1)
    if games:
        for game in games:
            print_game_from_csv(game, show_steps=True)
