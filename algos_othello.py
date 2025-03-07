import random
from operator import itemgetter
import numpy as np
import othello as oth
from collections import defaultdict
import math

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


# https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
# https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
class MCTS:

    def __init__(self, player):
        self.player = player
        self.w = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = 1.4142  # about sqrt(2)
        self.game = oth.Othello()

    def choose(self, node):

        if self.game.is_game_over():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            # find children
            valid_moves = self.game.get_valid_moves(node)
            for move in valid_moves:
                new_board = node.board.copy()
                self.game.apply_move(move, self.player)
                self.children[node].append(self.game.board)
                self.game.board = new_board
            return random.choice(self.children[node])
            #return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.w[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
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
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        # find children
        valid_moves = self.game.get_valid_moves(node)
        for move in valid_moves:
            new_board = node.board.copy()
            self.game.apply_move(move, self.player)
            self.children[node].append(self.game.board)
            self.game.board = new_board
        # self.children[node] = node.find_children()

    def _simulate(self, node):

        while True:
            if self.game.is_game_over():
                reward = oth.evaluate_board(node)
                return reward
            valid_moves = self.game.get_valid_moves(node)
            for move in valid_moves:
                new_board = node.board.copy()
                self.game.apply_move(move, self.player)
                self.children[node].append(self.game.board)
                self.game.board = new_board
            node = random.choice(self.children[node])
            # node = node.find_random_child()

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.w[node] += reward
            reward = - reward  # ??

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.w[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


def monte_carlo_play(board, player):
    tree = MCTS(player)
    game = oth.Othello()
    game.board = board.copy()
    if game.is_game_over():
        print ("over")
        return

    for _ in range (10000):
        tree.do_rollout(game.board)
    best_move = tree.choose(game.board)
    return best_move


def user_ai(board, player):
    return monte_carlo_play(board, player)