import numpy as np
import othello as oth

# Valeurs des cases
quadrant1 = np.array([[500, -150, 30,   10],
                       [-150, -250, 0,   0],
                       [30,    0,   1,   2],
                       [10,    0,   2,   16]])
quadrant2 = np.flip(quadrant1, axis=0)
quadrant3 = np.flip(quadrant1, axis=1)
quadrant4 = np.flip(quadrant2, axis=1)
CELL_VALUES = np.concatenate((np.concatenate([quadrant1, quadrant2], axis=0), np.concatenate([quadrant3, quadrant4], axis=0)), axis=1)
print(CELL_VALUES)

# 1.Minimax amélioré
DEPTH_IMRPOVED = 6

def evaluate_board_improved(game, player):
    # combinaison linéaire des heuristiques
    board = game.board
    return difference_pieces(board) + cell_values(game, player) + nb_possible_moves(game, player)

# Critère 1: Différence entre les pièces des 2 joueurs
def difference_pieces(board):
    """Basic evaluation function: counts the number of pieces per player."""
    return np.sum(board == oth.WHITE) - np.sum(board == oth.BLACK)

# Critère 2: valeur de case
def cell_values(game, player):
    return np.sum(CELL_VALUES[game.board == player])

# Critère 3: Mobilité
def nb_possible_moves(board, player):
    return len(board.get_valid_moves(player))

# Copy of minimax in Othello.py with the new evaluating function
def minimax_improved(board, depth, maximizing, player):
    """Minimax AI with depth limit."""
    game = oth.Othello()
    game.board = board.copy()

    if depth == 0 or game.is_game_over():
        return evaluate_board_improved(game, player), None

    valid_moves = game.get_valid_moves(player)
    best_move = None

    if maximizing:
        max_eval = float("-inf")
        for move in valid_moves:
            new_board = game.board.copy()
            game.apply_move(move, player)
            eval_score, _ = minimax_improved(new_board, depth - 1, False, -player)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in valid_moves:
            new_board = game.board.copy()
            game.apply_move(move, player)
            eval_score, _ = minimax_improved(new_board, depth - 1, True, -player)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
        return min_eval, best_move

# Improved minimax -- in streamlit has to become user_ai(board, player)
def improved_minimax_ai(board, player):
    _, best_move = minimax_improved(board, DEPTH_IMRPOVED, True, player)
    return best_move

# 2.Alpha-Beta Pruning
DEPTH_ALPHA_BETA = 7

