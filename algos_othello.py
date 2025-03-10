import random
from operator import itemgetter
import numpy as np
import othello as oth
from collections import defaultdict
import math
import copy
import time

# Temps d'exécution
times = []

# Valeurs des cases
quadrant1 = np.array([[500, -150, 30,   10],
                       [-150, -250, 0,   0],
                       [30,    0,   1,   2],
                       [10,    0,   2,   16]])
quadrant2 = np.flip(quadrant1, axis=0)
quadrant3 = np.flip(quadrant1, axis=1)
quadrant4 = np.flip(quadrant2, axis=1)
CELL_VALUES = np.concatenate((np.concatenate([quadrant1, quadrant2], axis=0), np.concatenate([quadrant3, quadrant4], axis=0)), axis=1)


#------ 1 . MINIMAX AMÉLIORÉ --------
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
    start = time.time()
    _, best_move = minimax_improved(board, DEPTH_IMPROVED, True, player)
    end = time.time()
    duration = end - start
    times.append(duration)
    print(duration)
    return best_move

#------ 2 . ALPHA-BETA PRUNING --------

# SOURCE: https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/
# SOURCE HEURISTIQUES (PSEUDOCODE) : https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/miniproject1_vaishu_muthu/Paper/Final_Paper.pdf

DEPTH_ALPHA_BETA = 7

def a_b_eval(game, player):
    # evaluate by adding mobility to combi lin
    board = game.board

    # hardcode les joueurs bc min & max
    p_mobility = len(game.get_valid_moves(1))
    o_mobility = len(game.get_valid_moves(-1))
    actual_m = 0
    if (p_mobility + o_mobility) != 0:
        actual_m = 100 * (p_mobility - o_mobility) / (p_mobility + o_mobility)
    return actual_m + difference_pieces(board) + cell_values(game, player) + nb_possible_moves(game, player)

#  Potential mobility calculations
def count_empty_spaces(board, player):
    p_value = player
    o_value = -player

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    potential_m = 0

    for r in range(8):
        for c in range(8):
            if board[r, c] == 0:
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == o_value:
                        potential_m += 1
                        break

    return potential_m

def a_b_eval2(game, player):
    # evaluate by adding potential mobility to combi lin
    board = game.board

    p_mobility = count_empty_spaces(board, player)
    o_mobility = count_empty_spaces(board, -player)
    potential_m = 0
    if (p_mobility + o_mobility) != 0:
        potential_m = 100 * (p_mobility - o_mobility) / (p_mobility + o_mobility)
    return potential_m + difference_pieces(board) + cell_values(game, player) + nb_possible_moves(game, player)

# ALPHA-BETA AVEC KILLER HEURISTIC ET ORIGINAL EVAL FUNCTION
def alpha_beta_pruning(board, depth, alpha, beta, maximizing, player, killer_moves):

    game = oth.Othello()
    game.board = board.copy()

    # if leaf or no valid moves (no children) return the board aka value of the curr position
    if depth == 0 or game.is_game_over():
        return evaluate_board_improved(game, player), None # TODO: temp eval function

    valid_moves = game.get_valid_moves(player)
    best_move = None

    k_d = killer_moves[depth]

    if maximizing:
        best = float("-inf")
        # look at killer moves first
        for move in k_d:
            if move in valid_moves: # explore
                val, _ = alpha_beta_pruning(board, depth - 1, alpha, beta, False, -player, killer_moves)
                best = max(best, val)
                alpha = max(alpha, best)
                best_move = move
                if beta <= alpha:
                    if move not in k_d:
                        if len(k_d) >= 2:
                            k_d.pop()  # Keep only the top 2 killer moves
                        k_d.insert(0, move)
                    break  # élaguer la branche et toutes les prochaines

        for move in valid_moves:
            if move in k_d:
                continue
            val, _ = alpha_beta_pruning(board, depth-1, alpha, beta, False, -player, killer_moves)
            best = max(best, val)
            alpha = max(alpha, best)
            best_move = move
            if beta <= alpha:
                if move not in k_d:
                    if len(k_d) >= 2:
                        k_d.pop()  # Keep only the top 2 killer moves
                    k_d.insert(0, move)
                break
        return best, best_move
    else:
        best = float("inf")

        for move in k_d:
            if move in valid_moves:
                val, _ = alpha_beta_pruning(board, depth - 1, alpha, beta, True, -player, killer_moves)
                best = min(best, val)
                beta = min(beta, best)
                best_move = move
                if beta <= alpha:
                    if move not in k_d:
                        if len(k_d) >= 2:
                            k_d.pop()  # Keep only the top 2 killer moves
                        k_d.insert(0, move)
                    break  # élaguer la branche et toutes les prochaines


        for move in valid_moves:
            if move in k_d:
                continue
            val, _ = alpha_beta_pruning(board, depth-1, alpha, beta, True, -player, killer_moves)
            best = min(best, val)
            beta = min(beta, best)
            best_move = move
            if beta <= alpha:
                if move not in k_d:
                    if len(k_d) >= 2:
                        k_d.pop()  # Keep only the top 2 killer moves
                    k_d.insert(0, move)
                break

        return best, best_move

# ALPHA-BETA SANS KILLER HEURISTIC, AVEC MOBILITÉ (potentielle) DANS L'ÉVALUATION
def alpha_beta_pruning_no_killer(board, depth, alpha, beta, maximizing, player):

    game = oth.Othello()
    game.board = board.copy()

    # if leaf or no valid moves (no children) return the board aka value of the curr position
    if depth == 0 or game.is_game_over():
        return a_b_eval2(game,player), None

    valid_moves = game.get_valid_moves(player)
    best_move = None

    if maximizing:
        best = float("-inf")
        for move in valid_moves:
            val, _ = alpha_beta_pruning_no_killer(board, depth-1, alpha, beta, False, -player)
            best = max(best, val)
            alpha = max(alpha, best)
            best_move = move
            if beta <= alpha:
                break
        return best, best_move
    else:
        best = float("inf")

        for move in valid_moves:
            val, _ = alpha_beta_pruning_no_killer(board, depth-1, alpha, beta, True, -player)
            best = min(best, val)
            beta = min(beta, best)
            best_move = move
            if beta <= alpha:
                break

        return best, best_move

# Contenu de "user_ai"
def alpha_beta_ai(board, player):
    start = time.time()
    killer_moves = dict()
    for i in range(DEPTH_ALPHA_BETA+1):
        killer_moves[i] = []
    _, best_move = alpha_beta_pruning(board, DEPTH_ALPHA_BETA, float("-inf"), float("inf"), True, player, killer_moves)
    end = time.time()
    duration = end - start
    times.append(duration)
    print(duration)
    return best_move

# Contenu de "user_ai"
def alpha_beta_ai_2(board, player):
    start = time.time()
    _, best_move = alpha_beta_pruning_no_killer(board, DEPTH_ALPHA_BETA, float("-inf"), float("inf"), True, player)
    end = time.time()
    duration = end - start
    times.append(duration)
    print(duration)
    return best_move

# ---- 3 . MCTS ----

LIMIT_EXPLORATIONS = 1000

# https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
# https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
class MCTS:

    def __init__(self, player):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = 1
        self.player = player

        # node is an instance of Othello

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_game_over():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            possible_moves = node.get_valid_moves(self.player)
            new_board = copy.deepcopy(node.board)
            move = random.choice(possible_moves)
            game = oth.Othello()
            game.board = new_board
            game.apply_move(move, self.player)
            # print("game board \n", game.board)
            return game
            #return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        # print("max \n", max(self.children[node], key=score).board)
        return max(self.children[node], key=score)

    def do_rollout(self, node):
        # Make the tree one layer better. (Train for one iteration.)
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        # Find an unexplored descendent of `node`
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        #Update the `children` dict with the children of `node`
        if node in self.children:
            return  # already expanded
        self.children[node] = []
        possible_moves = node.get_valid_moves(self.player)

        new_board_1 = node.board.copy()
        for move in possible_moves:
            # print("new board \n", new_board_1)
            game = oth.Othello()
            game.board = new_board_1.copy()
            # print("before move \n", new_board_1, game.board)
            game.apply_move(move, self.player)
            # print("children appended \n", game.board)
            self.children[node].append(game)
        #self.children[node] = node.find_children()

    def _simulate(self, node):
        # Returns the reward for a random simulation (to completion) of `node`
        invert_reward = True
        while True:
            if node.is_game_over():
                reward = oth.evaluate_board(node.board)
                return 1 - reward if invert_reward else reward
            possible_moves = node.get_valid_moves(self.player)
            new_board = copy.deepcopy(node.board)

            if possible_moves is None or len(possible_moves)==0:
                reward = oth.evaluate_board(node.board)
                return 1 - reward if invert_reward else reward
            move = random.choice(possible_moves)
            game = oth.Othello()
            game.board = new_board
            game.apply_move(move, self.player)
            node = game

            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        # Send the reward back up to the ancestors of the leaf
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        # Select a child of node, balancing exploration & exploitation

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            # Upper confidence bound for trees
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


def monte_carlo_play(board, player):
    start = time.time()
    tree = MCTS(player)
    game = oth.Othello()
    game.board = board.copy()
    if game.is_game_over():
        print ("over")
        return

    for _ in range (LIMIT_EXPLORATIONS):
        tree.do_rollout(game)
    best_board = (tree.choose(game)).board
    diff = best_board - game.board
    # print("best board \n", best_board)
    # print("diff \n", diff)

    # trouver la position avec un 1
    position = np.argwhere(abs(diff)==1)
    # print(position)
    best_move = (position[0][0], position[0][1])
    # print("best move \n", best_move)
    end = time.time()
    duration = end - start
    times.append(duration)
    print(duration)
    return best_move

# ----- USER_AI ------

def user_ai(board, player):
    return alpha_beta_ai(board, player)