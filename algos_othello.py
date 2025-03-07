import random
from operator import itemgetter
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
DEPTH_IMPROVED = 6

def improved_minimax_ai(board, player):
    _, best_move = minimax_improved(board, DEPTH_IMPROVED, True, player)
    return best_move

# 2.Alpha-Beta Pruning

# https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/
def alpha_beta_pruning(board, depth, alpha, beta, maximizing, player):
    game = oth.Othello()
    game.board = board.copy()

    # if leaf or no valid moves (no children) return the board aka value of the curr position
    if depth == 0 or game.is_game_over():
        return evaluate_board_improved(game, player), None # TODO: temp eval function

    valid_moves = game.get_valid_moves(player)
    best_move = None

    if maximizing:
        best = float("-inf")
        for move in valid_moves:
            val, _ = alpha_beta_pruning(board, depth-1, alpha, beta, False, -player)
            best = max(best, val)
            alpha = max(alpha, best)
            best_move = move
            if beta <= alpha:
                break # élaguer la branche et toutes les prochaines
        return best, best_move
    else:
        best = float("inf")
        for move in valid_moves:
            val, _ = alpha_beta_pruning(board, depth-1, alpha, beta, True, -player)
            best = min(best, val)
            beta = min(beta, best)
            best_move = move
            if beta <= alpha:
                break
        return best, best_move

# Paste on the platform
DEPTH_ALPHA_BETA = 7

def alpha_beta_ai(board, player):
    _, best_move = alpha_beta_pruning(board, DEPTH_ALPHA_BETA, float("-inf"), float("inf"), True, player)
    return best_move

def user_ai(board, player):
    return monte_carlo(board, player)


# https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/
LIMIT_EXPLORATIONS = 10000
def monte_carlo(board, player):
    game = oth.Othello()
    game.board = board.copy()
    possible_nodes = [[board,0]]

    best_score = 0
    best_move = None


    for _ in range(LIMIT_EXPLORATIONS):

        # select root node
        root = max(possible_nodes, key=itemgetter(1))[0]
        game.board = root
        valid_moves = game.get_valid_moves(player)

        for i in range(len(valid_moves)):
            new_board = game.board.copy()
            game.apply_move(valid_moves[i], player)

            # play until game over
            score = play_random(new_board, player)
            if score > best_score:
                best_score = score
                best_move = valid_moves[i]
            print(score)
            possible_nodes.append([new_board, score])

    return best_move



def play_random(board, player):
    game = oth.Othello()
    game.board = board.copy()
    print("is game over?", game.is_game_over(), "\n", game.board)
    if game.is_game_over() :
        score = np.sum(game.board == oth.WHITE)- np.sum(game.board == oth.BLACK)
        print("good score ", score)
        return score

    valid_moves = game.get_valid_moves(player)
    move = random.choice(valid_moves)
    game.apply_move(move, player)
    score = play_random(game.board, player)
    print(score)
    return score


