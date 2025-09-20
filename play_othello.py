def print_board(board):
    symbols = {0: ".", 1: "B", 2: "W"}
    for r in range(8):
        print(" ".join(symbols[board[r][c]] for c in range(8)))
    print()


def count_pieces(board):
    black_count = sum(row.count(1) for row in board)
    white_count = sum(row.count(2) for row in board)
    return black_count, white_count


def position_to_coord(pos):
    """将棋盘坐标转换为行列索引"""
    col, row = pos[0], pos[1]
    row = int(row) - 1  # 转换为 0-7 行索引
    col = ord(col) - ord('a')  # 转换为 0-7 列索引
    return row, col


def inside(r, c):
    return 0 <= r < 8 and 0 <= c < 8


def apply_move(board, pos, player):
    r, c = position_to_coord(pos)
    if board[r][c] != 0:
        return False  # 已经有子了，不合法

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
        return False  # 没有翻子，不合法

    # 落子并翻转
    board[r][c] = player
    for fr, fc in flipped:
        board[fr][fc] = player
    return True


def play_othello(moves):
    # 初始化棋盘
    board = [[0] * 8 for _ in range(8)]
    board[3][3], board[3][4] = 2, 1  
    board[4][3], board[4][4] = 1, 2  

    print("Initial board:")
    print_board(board)

    player = 1  # 黑先手
    skip_positions = {27, 28, 35, 36}  # 初始子位置

    for i, pos in enumerate(moves):
        print(f"Move {i+1}: {'Black' if player==1 else 'White'} -> {pos}")
        if not apply_move(board, pos, player):
            player = 3 - player

        apply_move(board, pos, player)
        print_board(board)
        player = 3 - player


# 你的新的数据集
moves = ['f5', 'd6', 'c4', 'd3', 'e6', 'f4', 'e3', 'f6', 'c5', 'b4', 'e7', 'f3', 'c6', 
         'd7', 'b5', 'a5', 'c3', 'b3', 'g5', 'h5', 'g4', 'h4', 'e2', 'g6', 'b6', 'd8', 
         'c7', 'c8', 'a4', 'a6', 'a7', 'f1', 'a3', 'c2', 'd2', 'b2', 'e1', 'b7', 'g3', 
         'h3', 'f2', 'd1', 'a1', 'a2', 'b1', 'a8', 'c1', 'g1', 'f7', 'g8', 'e8', 'f8', 
         'b8', 'g7', 'h8', 'h7', 'h6', 'h2', 'g2', 'h1']

play_othello(moves)
