import streamlit as st
import numpy as np
import random

# Définition des constantes
EMPTY = 0
BLACK = 1
WHITE = -1

# Classe du jeu Othello
class Othello:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3, 3], self.board[4, 4] = WHITE, WHITE
        self.board[3, 4], self.board[4, 3] = BLACK, BLACK
        self.current_player = BLACK

    def get_valid_moves(self, player):
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(row, col, player):
                    valid_moves.append((row, col))
        return valid_moves

    def is_valid_move(self, row, col, player):
        if self.board[row, col] != EMPTY:
            return False
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            flipped = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == -player:
                flipped.append((r, c))
                r += dr
                c += dc
            if flipped and 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
                return True
        return False

    def apply_move(self, move, player):
        row, col = move
        self.board[row, col] = player
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            flipped = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == -player:
                flipped.append((r, c))
                r += dr
                c += dc
            if flipped and 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
                for fr, fc in flipped:
                    self.board[fr, fc] = player

# Fonction IA simple (Minimax aléatoire)
def ai_move(board, player):
    valid_moves = [move for move in Othello().get_valid_moves(player)]
    return random.choice(valid_moves) if valid_moves else None

# Interface Streamlit
st.title("Othello - Jouez contre l'IA")

game = Othello()

# Affichage du plateau
st.write("Plateau de jeu:")
st.write(game.board)

# Choix de mode
mode = st.selectbox("Mode de jeu", ["Humain vs IA", "IA vs IA"])

if mode == "Humain vs IA":
    row = st.number_input("Choisissez une ligne (0-7)", min_value=0, max_value=7, step=1)
    col = st.number_input("Choisissez une colonne (0-7)", min_value=0, max_value=7, step=1)
    if st.button("Jouer"):  # Tour du joueur humain
        if (row, col) in game.get_valid_moves(BLACK):
            game.apply_move((row, col), BLACK)
            ai_play = ai_move(game.board, WHITE)
            if ai_play:
                game.apply_move(ai_play, WHITE)
            st.write("Mouvement de l'IA :", ai_play)
            st.write("Nouveau plateau:")
            st.write(game.board)
        else:
            st.write("Coup invalide. Essayez encore.")

elif mode == "IA vs IA":
    if st.button("Lancer la partie IA vs IA"):
        while game.get_valid_moves(game.current_player):
            ai_play = ai_move(game.board, game.current_player)
            if ai_play:
                game.apply_move(ai_play, game.current_player)
                game.current_player = -game.current_player
        st.write("Partie terminée! Voici le plateau final:")
        st.write(game.board)
