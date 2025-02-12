import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time  # Import pour ajouter un délai entre les coups

# Définition des constantes
EMPTY = 0
BLACK = 1
WHITE = -1
DEPTH = 3  # Profondeur du Minimax

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

def update_leaderboard(student_id, final_score):
    df = st.session_state.leaderboard
    df["ID"] = df["ID"].astype(int)
    df["Score"] = df["Score"].astype(int)
    # 🔥 Mise à jour du leaderboard
    student_id = int(student_id)
    existing_index = df.index[df["ID"] == student_id].tolist()


    if existing_index:
        current_best_score = df.at[existing_index[0], "Score"]
        if final_score > current_best_score:
            df.at[existing_index[0], "Score"] = final_score
            st.success(f"🎉 Félicitations {student_id} ! Votre score a été amélioré de {current_best_score} à {final_score}.")
        else:
            st.info(f"📌 Votre score actuel ({final_score}) n'a pas dépassé votre meilleur score ({current_best_score}).")
    else:
        # Ajouter un nouvel ID avec son score
        new_entry = pd.DataFrame([[student_id, final_score]], columns=["ID", "Score"])
        df = pd.concat([df, new_entry], ignore_index=True)

    # Trier et sauvegarder
    df = df.sort_values(by="Score", ascending=False)
    st.session_state.leaderboard = df
    save_leaderboard(df)

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

# Minimax AI
def evaluate_board(board):
    """Basic evaluation function: counts the number of pieces per player."""
    return np.sum(board == WHITE) - np.sum(board == BLACK)

def minimax(board, depth, maximizing, player):
    """Minimax AI with depth limit."""
    game = Othello()
    game.board = board.copy()

    if depth == 0 or game.is_game_over():
        return evaluate_board(game.board), None

    valid_moves = game.get_valid_moves(player)
    best_move = None

    if maximizing:
        max_eval = float("-inf")
        for move in valid_moves:
            new_board = game.board.copy()
            game.apply_move(move, player)
            eval_score, _ = minimax(new_board, depth - 1, False, -player)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in valid_moves:
            new_board = game.board.copy()
            game.apply_move(move, player)
            eval_score, _ = minimax(new_board, depth - 1, True, -player)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
        return min_eval, best_move

def minimax_ai(board, player):
    """AI wrapper for Minimax with depth=3"""
    _, best_move = minimax(board, DEPTH, True, player)
    return best_move

# Interface Streamlit
st.title("🏆 Othello - Compétition TP1 ift3335 !")

st.title("Compétition entre IA !")

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

            if st.button("Lancer la partie IA vs Minimax AI"):
                game = Othello()

                # Préparer une seule figure pour tout le jeu
                fig, ax = plt.subplots(figsize=(8, 8))
                plot_placeholder = st.empty()  # Réserve l'espace pour la figure

                while not game.is_game_over():
                    valid_moves = game.get_valid_moves(game.current_player)
                    if not valid_moves:
                        game.current_player = -game.current_player
                        continue

                    # AI joue
                    current_ai = user_ai if game.current_player == BLACK else minimax_ai
                    move = current_ai(game.board, game.current_player)

                    if move:
                        game.apply_move(move, game.current_player)

                    # Mettre à jour le plateau dans la même figure
                    ax.clear()
                    ax.set_facecolor("#006400")  
                    
                    for x in range(9):
                        ax.plot([x-0.5, x-0.5], [-0.5, 7.5], color='black', linewidth=2)
                        ax.plot([-0.5, 7.5], [x-0.5, x-0.5], color='black', linewidth=2)
                    
                    for row in range(8):
                        for col in range(8):
                            if game.board[row, col] == BLACK:
                                ax.add_patch(plt.Circle((col, row), 0.4, color='black', zorder=2))
                            elif game.board[row, col] == WHITE:
                                ax.add_patch(plt.Circle((col, row), 0.4, color='white', zorder=2, edgecolor="black", linewidth=2))
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(-0.5, 7.5)
                    ax.set_ylim(7.5, -0.5)

                    plot_placeholder.pyplot(fig)  # Met à jour dans la même figure
                    time.sleep(1)  # Pause d'une seconde entre chaque coup

                    game.current_player = -game.current_player

                st.write(f"🎉 Partie terminée pour l'étudiant {student_id} !")

                final_score = np.sum(game.board == BLACK) - np.sum(game.board == WHITE)
                st.write(f"Votre score : {final_score}")

                # Mise à jour du leaderboard
                update_leaderboard(student_id, final_score)

                
        else:
            st.error("⚠️ Votre code doit définir une fonction `user_ai(board, player)`.")  

    except Exception as e:
        st.error(f"❌ Erreur dans votre code : {e}")

# Affichage du classement
st.subheader("🏅 Classement des étudiants")
st.session_state.leaderboard["ID"] = st.session_state.leaderboard["ID"].astype(int)
st.session_state.leaderboard["Score"] = st.session_state.leaderboard["Score"].astype(int)
st.dataframe(st.session_state.leaderboard)


# deux IA contre eux
st.subheader("🤖 IA 1")
id_ia1 = st.text_input("ID de l'étudiant IA 1")
code_ia1 = st.text_area("Entrez le code de IA 1 :", height=200)

st.subheader("🤖 IA 2")
id_ia2 = st.text_input("ID de l'étudiant IA 2")
code_ia2 = st.text_area("Entrez le code de IA 2 :", height=200)

if id_ia1 and id_ia2 and code_ia1 and code_ia2:
    try:
        # Initialisation des variables IA
        globals()["user_ai1"] = None
        globals()["user_ai2"] = None

        exec(code_ia1, globals())
        exec(code_ia2, globals())

        if "user_ai1" in globals() and "user_ai2" in globals() and callable(globals()["user_ai1"]) and callable(globals()["user_ai2"]):
            st.success(f"Les IA {id_ia1} et {id_ia2} ont été chargées !")

            if st.button("Démarrer la compétition ⚔️"):
                game = Othello()

                # Préparer une seule figure pour tout le jeu
                fig, ax = plt.subplots(figsize=(8, 8))
                plot_placeholder = st.empty()  # Réserve l'espace pour la figure

                # Affichage des mouvements
                while not game.is_game_over():
                    valid_moves = game.get_valid_moves(game.current_player)
                    if not valid_moves:
                        game.current_player = -game.current_player
                        continue

                    # Sélection de l'IA en fonction du joueur courant
                    current_ai = user_ai1 if game.current_player == BLACK else user_ai2
                    move = current_ai(game.board, game.current_player)

                    if move:
                        game.apply_move(move, game.current_player)

                    # Mettre à jour le plateau dans la même figure
                    ax.clear()
                    ax.set_facecolor("#006400")  

                    for x in range(9):
                        ax.plot([x-0.5, x-0.5], [-0.5, 7.5], color='black', linewidth=2)
                        ax.plot([-0.5, 7.5], [x-0.5, x-0.5], color='black', linewidth=2)

                    for row in range(8):
                        for col in range(8):
                            if game.board[row, col] == BLACK:
                                ax.add_patch(plt.Circle((col, row), 0.4, color='black', zorder=2))
                            elif game.board[row, col] == WHITE:
                                ax.add_patch(plt.Circle((col, row), 0.4, color='white', zorder=2, edgecolor="black", linewidth=2))

                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(-0.5, 7.5)
                    ax.set_ylim(7.5, -0.5)

                    plot_placeholder.pyplot(fig)  # Met à jour dans la même figure
                    time.sleep(1)  # Pause d'une seconde entre chaque coup

                    game.current_player = -game.current_player

                # Déterminer le gagnant
                score_ia1 = np.sum(game.board == BLACK)
                score_ia2 = np.sum(game.board == WHITE)

                if score_ia1 > score_ia2:
                    winner = id_ia1
                    st.success(f"🏆 L'IA {id_ia1} a gagné avec un score de {score_ia1} contre {score_ia2} !")
                elif score_ia2 > score_ia1:
                    winner = id_ia2
                    st.success(f"🏆 L'IA {id_ia2} a gagné avec un score de {score_ia2} contre {score_ia1} !")
                else:
                    st.warning(f"🤝 Match nul entre {id_ia1} et {id_ia2} ! Score : {score_ia1} - {score_ia2}")

        else:
            st.error("⚠️ Les codes doivent définir des fonctions `user_ai1(board, player)` et `user_ai2(board, player)`.")  

    except Exception as e:
        st.error(f"❌ Erreur dans les codes soumis : {e}")