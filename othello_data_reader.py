import csv
import json

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

def split_moves(move_string):
    """将走棋字符串分割成每步的位置"""
    moves = [move_string[i:i+2] for i in range(0, len(move_string), 2)]
    return moves

def get_move_players(moves):
    """确定每步走棋对应的玩家，考虑可能的跳过回合"""
    # 初始化棋盘
    board = [[0] * 8 for _ in range(8)]
    board[3][3], board[3][4] = 2, 1  # 初始布局
    board[4][3], board[4][4] = 1, 2
    
    current_player = 1  # 黑方(1)先手
    move_players = []
    
    for pos in moves:
        # 尝试当前玩家落子
        if apply_move([row.copy() for row in board], pos, current_player):
            # 落子有效，记录当前玩家并切换
            move_players.append(current_player)
            # 实际应用落子
            apply_move(board, pos, current_player)
            current_player = 3 - current_player  # 切换玩家
        else:
            # 落子无效，说明当前玩家跳过，由对方玩家落子
            current_player = 3 - current_player
            move_players.append(current_player)
            # 应用对方玩家的落子
            apply_move(board, pos, current_player)
            current_player = 3 - current_player  # 切换玩家
    
    return move_players

def read_othello_data(file_path):
    """读取Othello数据集并添加每步的玩家信息"""
    othello_data = []
    
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            print("数据集包含的字段:", csv_reader.fieldnames)
            print()
            
            for row in csv_reader:
                # 解析走棋步骤
                moves = split_moves(row['game_moves'])
                # 获取每步对应的玩家
                move_players = get_move_players(moves)
                
                # 构建游戏数据字典
                game_data = {
                    'eOthello_game_id': int(row['eOthello_game_id']),
                    'winner': int(row['winner']),
                    # 'game_moves': row['game_moves'],
                    # 'parsed_moves': moves,
                    'move_count': len(moves),
                    # 'move_players': move_players,  # 每步对应的玩家
                    'moves_with_players': list(zip(moves, move_players))  # 走棋与玩家的组合
                }
                othello_data.append(game_data)
                
                # 展示前2条数据作为示例
                if len(othello_data) <= 2:
                    print(f"游戏ID: {game_data['eOthello_game_id']}")
                    print(f"获胜者: {game_data['winner']} (1=黑方, 2=白方)")
                    print(f"总步数: {game_data['move_count']}")
                    print("前5步走法及玩家:")
                    for move, player in game_data['moves_with_players'][:5]:
                        print(f"  {move}: {'黑方' if player == 1 else '白方'}")
                    print("-" * 60)
        
        print(f"成功读取 {len(othello_data)} 条游戏数据")
        return othello_data
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
    return None

if __name__ == "__main__":
    # 替换为你的CSV文件路径
    csv_file_path = "othello_dataset.csv"
    
    # 读取数据
    othello_games = read_othello_data(csv_file_path)
    
    # 如果有数据，保存为JSON文件
    if othello_games:
        with open('othello_with_players.json', 'w', encoding='utf-8') as f:
            json.dump(othello_games, f, ensure_ascii=False, indent=2)
        print("\n数据已保存到 othello_with_players.json")
