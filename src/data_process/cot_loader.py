import json


def convert_board_to_positions(rows):
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    black_pieces = []
    white_pieces = []
    
    for row_idx, row in enumerate(rows):
        cells = row.split()
        
        for col_idx, cell in enumerate(cells):
            position = f"{columns[col_idx]}{row_idx + 1}"
            
            if cell == 'B':
                black_pieces.append(position)
            elif cell == 'W':
                white_pieces.append(position)
    
    return {
        "black_pieces": black_pieces,
        "white_pieces": white_pieces
    }


def format_othello_record_for_training(record: dict, mode='train') -> str:
    """
    Converts a structured Othello data record into a single string
    formatted with ChatML for supervised fine-tuning.

    Args:
        record: A dictionary containing the move data and CoT analysis.

    Returns:
        A single string ready for model training.
    """
    # 1. Extract necessary information from the input record
    board_state = record['board_state']
    rows = board_state.split('\n')
    # numbered_rows = [f"row {i+1} {row}" for i, row in enumerate(rows)]
    # board_state = '\n'.join(numbered_rows)
    board_state = convert_board_to_positions(rows)

    player_id = record['player']
    chosen_move = record['position']
    cot_data = record['cot']
    try:
        del cot_data['analysis']['opponent_response_prediction']
        del cot_data['analysis']['alternatives_considered']
        del cot_data['analysis']['move_justification']['capture_details']
    except:
        pass
    prev_action = record['prev_action']

    # Convert player ID to a human-readable string
    player_str = "Black (B)" if player_id == 1 else "White (W)"

    # 2. Define the prompt components based on the ChatML format
    
    # The system prompt sets the model's persona and overall goal.
    system_prompt = "You are an Othello grandmaster. Your task is to analyze a given board state and a specific move, then provide a structured thought process in JSON format that explains the strategic rationale behind that move."

    # The user prompt provides the specific context for the current turn.
    if mode == 'train':
        user_prompt = (
            f"prev action:{prev_action}\n"
            f"Player to move: {player_str}\n"
            f"Chosen Move: {chosen_move.upper()}\n\n"
            f"Board State:\n"
            f"{board_state}\n\n"
            f"Please provide a detailed analysis justifying why the chosen move is a good one."
        )

    elif mode == 'eval':
        user_prompt = (
            f"prev action:{prev_action}\n"
            f"Player to move: {player_str}\n"
            f"Board State:\n"
            f"{board_state}\n\n"
            f"Please provide a detailed analysis justifying why the chosen move is a good one."
        )
    
    else:
        assert false

    # The assistant's response is the "ground truth" the model learns to generate.
    # We convert the CoT dictionary into a clean, indented JSON string.
    assistant_response = json.dumps(cot_data, indent=2, ensure_ascii=False)

    # 3. Combine everything into the final formatted string
    if mode == 'train':
        return (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{user_prompt}\n"
            f"<|assistant|>\n{assistant_response}"
        )
    elif mode == 'eval':
        return (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{user_prompt}\n"
        )


def load_cot_data(file_path, mode='train'):
    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    cot_data = []
    for game in raw_data:
        prev_action = []
        for step in game["cot_analysis"]:
            if step["cot_format"] == "valid_json":
                step['prev_action'] = prev_action
                
                cot_data.append(format_othello_record_for_training(step, mode=mode))
            
            prev_action.append(step['position'])
            prev_action = prev_action[-5:]

    return cot_data
        
if __name__ == '__main__':
    load_cot_data('/data/data_public/zjy/Othello-Qwen/data/othello_with_cot.json')
    