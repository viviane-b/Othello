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
| Alpha-Beta            | 20             | 3.84                                           |
| Recherche Monte-carlo | 40             | 0.74                                           |

Puisque les algorithmes minimax et Alpha-Beta ne contiennent pas d'aléatoire, si
on ne modifie pas l'algorithme, à chaque fois qu'on lance une nouvelle partie contre
minimax de base, on obtient le même score. Par contre, avec Monte-Carlo, ça peut changer
entre chaque partie. Par exemple, en faisant quelques parties, le meilleur score que
nous avons obtenu avec Monte-Carlo est de 40, mais nous avons aussi obtenu un score de -6 
à une autre partie. 