# Jeu d'Othello - Rapport


### Algorithmes implémentés et heuristiques utilisées

#### Minimax amélioré
Le minimax amélioré a été implémenté en utilisant plus de plis, donc une profondeur 
de graphe plus élevée pour évaluer plus de coups en avance. Nous avons utilisé 3 heuristiques
pour évaluer les noeuds terminaux: la différence entre le nombre de pièces noires vs blanches
sur le jeu, la valeur des cases sur lesquelles nos jetons sont placés (à partir des valeurs des 
cases données dans les instructions), et le nombre de positions
où le joueur peut jouer. Ensuite, nous avons effectué une combinaison linéaire des 3 critères
pour obtenir un score de chaque noeud terminal pour pouvoir les comparer.
La combinaison linéaire est simplement l'addition de la valeur de chaque critère.



#### Alpha-Beta
Nous avons deux implémentations de la recherche alpha-beta en se basant sur [ce papier](https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/miniproject1_vaishu_muthu/Paper/Final_Paper.pdf)

Dans la première, on a gardé le comportement habituel de l'algorithme (tiré de GeeksForGeeks) et 
nous avons modifié la fonction d'évaluation employée dans minimax amélioré pour qu'elle considère également 
la mobilité potentielle de notre joueur (-> a_b_eval2). Cette dernière indique le nombre de cases vides adjacentes à l'adversaire,
elle donne le nombre de déplacements possibles du joueur à travers les prochains tours du jeu. On l'a chosie par rapport
à la mobilité actuelle, car le fait qu'elle regarde "dans le futur" la rend plus intéressante. De plus, parmi 
les heuristiques du papier duquel nous nous sommes inspirées, elle était la plus intuitive à implémenter. 
Nous avons également implémenté la mobilité actuelle (a_b_eval), mais il s'avère que le score est le même dans les deux cas, 
seule la complexité change pour la mobilité potentielle, car le calcul est plus long.

La deuxième façon dont nous avons implémenté l'algorithme fut avec l'usage du "killer heuristic". Son but est de prioriser
les choix ayant causé un élagage à un niveau donné ("killer"), car ces choix pourraient être bons dans un autre sous-arbres et ne pas causer un
élagage. Son but est donc de gagner du temps au lieu d'attendre pour se rendre à un bon sous-arbre. Ici, on a utilisé
la même fonction d'évaluation que minimax (evaluation_board_improved) pour voir si le comportement est vraiment différent.

Le score est le même dans tous les cas (actual mobility, potential mobility, killer heuristic), mais le temps d'exécution est 
un peu différent. 


#### Recherche arborescente Monte-Carlo
Pour la recherche Monte-Carlo, nous nous sommes inspirées d'un [programme](https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1)
trouvé en ligne qui implémentait une recherche Monte-Carlo générale,
et nous l'avons adaptée au jeu d'Othello. Nous avons utilisé une valeur de poids
d'exploration de 1 dans la formule de UCB1. Cet algorithme n'utilise pas d'heuristiques.



### Comparaison entre les algorithmes
Voici les performances obtenues avec chaque algorithme:

| Algorithme            | Meilleur score | Temps d'exécution moyen par coup (en secondes) |
|-----------------------|----------------|------------------------------------------------|
| Minimax amélioré      | 22             | 13.03                                          |
| Alpha-Beta            | 20             | 0.28                                           |
| Recherche Monte-carlo | 40             | 0.74                                           |

Puisque les algorithmes minimax et Alpha-Beta ne contiennent pas d'aléatoire, si
on ne modifie pas l'algorithme, à chaque fois qu'on lance une nouvelle partie contre
minimax de base, on obtient le même score. Par contre, avec Monte-Carlo, ça peut changer
entre chaque partie. Par exemple, en faisant quelques parties, le meilleur score que
nous avons obtenu avec Monte-Carlo est de 40, mais nous avons aussi obtenu un score de -6 
à une autre partie. 