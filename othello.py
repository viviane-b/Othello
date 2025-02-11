import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os

# Définition des constantes
EMPTY = 0
BLACK = 1
WHITE = -1

# 📌 Chemin du fichier où sauvegarder les scores
LEADERBOARD_FILE = "leaderboard.csv"

# 📌 Charger le leaderboard depuis un fichier CSV au démarrage
def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        return pd.read_csv(LEADERBOARD_FILE)
    return pd.DataFrame(columns=["ID", "Score"])

# 📌 Sauvegarder le leaderboard dans un fichier CSV
def save_leaderboard(df):
    df.to_csv(LEADERBOARD_FILE, index=False)

# Charger les scores existants au démarrage
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = load_leaderboard()

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
        if move is None:
            return
        row, col = move
        self.board[row, col] = player  # Place la pièce
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
                    self.board[fr, fc] = player  # Retourne les pièces

    def is_game_over(self):
        return not self.get_valid_moves(BLACK) and not self.get_valid_moves(WHITE)

# Affichage du plateau
def draw_board(board):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#006400")  # Fond vert foncé

    for x in range(9):
        ax.plot([x-0.5, x-0.5], [-0.5, 7.5], color='black', linewidth=2)
        ax.plot([-0.5, 7.5], [x-0.5, x-0.5], color='black', linewidth=2)

    for row in range(8):
        for col in range(8):
            if board[row, col] == BLACK:
                ax.add_patch(plt.Circle((col, row), 0.4, color='black', zorder=2))
            elif board[row, col] == WHITE:
                ax.add_patch(plt.Circle((col, row), 0.4, color='white', zorder=2, edgecolor="black", linewidth=2))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)

    st.pyplot(fig)

# Interface Streamlit
st.title("🏆 Othello - Compétition des Étudiants !")

# Formulaire pour entrer l'ID étudiant
student_id = st.text_input("Entrez votre ID étudiant")

st.write("Soumettez votre propre IA sous forme de fonction Python.")

# Champ de soumission de code
user_code = st.text_area("Entrez votre code Python ici :", height=200)

if student_id and user_code:
    try:
        exec(user_code, globals())

        if "user_ai" in globals() and callable(globals()["user_ai"]):
            user_ai = globals()["user_ai"]
            st.success(f"Votre IA a été enregistrée pour l'étudiant {student_id} !")

            if st.button("Lancer la partie IA vs IA"):
                game = Othello()

                while not game.is_game_over():
                    valid_moves = game.get_valid_moves(game.current_player)
                    if not valid_moves:
                        game.current_player = -game.current_player
                        continue

                    current_ai = user_ai if game.current_player == BLACK else lambda b, p: random.choice(valid_moves)
                    move = current_ai(game.board, game.current_player)

                    if move:
                        game.apply_move(move, game.current_player)
                    game.current_player = -game.current_player

                st.write(f"🎉 Partie terminée pour l'étudiant {student_id} !")

                final_score = np.sum(game.board == BLACK) - np.sum(game.board == WHITE)
                st.write(f"Votre score : {final_score}")

                # 🔥 Mise à jour du leaderboard
                existing_entry = st.session_state.leaderboard[st.session_state.leaderboard["ID"] == student_id]

                if not existing_entry.empty:
                    current_best_score = existing_entry["Score"].values[0]
                    if final_score > current_best_score:
                        st.session_state.leaderboard.loc[st.session_state.leaderboard["ID"] == student_id, "Score"] = final_score
                else:
                    new_entry = pd.DataFrame([[student_id, final_score]], columns=["ID", "Score"])
                    st.session_state.leaderboard = pd.concat([st.session_state.leaderboard, new_entry], ignore_index=True)

                # Trier et sauvegarder
                st.session_state.leaderboard = st.session_state.leaderboard.sort_values(by="Score", ascending=False)
                save_leaderboard(st.session_state.leaderboard)

                draw_board(game.board)

        else:
            st.error("⚠️ Votre code doit définir une fonction `user_ai(board, player)`.")  

    except Exception as e:
        st.error(f"❌ Erreur dans votre code : {e}")

# Affichage du classement
st.subheader("🏅 Classement des étudiants")
st.dataframe(st.session_state.leaderboard)
