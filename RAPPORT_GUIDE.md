# Guide de Rédaction du Rapport – Colorisation d'Images en Machine Learning

Ce guide a été spécialement conçu pour vous accompagner dans la rédaction de votre rapport PDF. Tous les concepts, choix techniques et formules mathématiques y sont expliqués de manière claire et vulgarisée, avec des arguments solides prêts à être intégrés dans votre devoir.

---

## Plan Recommandé pour Votre Rapport

1. **Introduction & Formulation du Problème**
2. **Représentation des Données : Pourquoi l'Espace CIELab ?**
3. **Architecture U-Net : Principes & Rôle des Skip-Connections**
4. **Première Approche : U-Net + MSE & Analyse Critique des Résultats**
5. **Méthodologie d'Évaluation : Métriques Quantitatives ($\Delta ab$ et Dispersion)**
6. **Amélioration : Transfert d'Apprentissage avec ResNet-18**
7. **Comparaison Expérimentale & Discussion**
8. **Conclusion & Perspectives**
9. **Déclaration d'Utilisation de l'IA Générative (Obligatoire)**

---

## 1. Introduction & Formulation du Problème

- **Objectif** : Transformer une image monochrome (niveaux de gris) en une image couleur plausible et naturelle.
- **Difficulté fondamentale (Multimodalité)** : La colorisation est un problème **mal posé** et **ambigu**. Pour une même fleur en niveaux de gris, plusieurs couleurs réelles sont possibles (rouge, jaune, violet, blanc...). Il n'existe donc pas une seule "bonne" réponse universelle.

---

## 2. Choix de l'Espace CIELab vs RGB

### Pourquoi ne pas utiliser le format standard RGB ?
- En RGB, chaque pixel est défini par $(R, G, B)$.
- Les trois canaux contiennent à la fois l'intensité lumineuse et la couleur. Si l'on fournit une image en noir et blanc à un réseau pour prédire $(R, G, B)$, le réseau doit réapprendre à prédire la luminosité de l'image (qu'il avait pourtant déjà en entrée).

### L'avantage décisif de l'espace CIELab :
- **Canal $L$ (Luminance)** : varie de 0 (noir) à 100 (blanc). Il représente fidèlement la structure, les contours et la luminosité de l'image.
- **Canal $a$** : axe chromatique vert (négatif) $\leftrightarrow$ rouge (positif).
- **Canal $b$** : axe chromatique bleu (négatif) $\leftrightarrow$ jaune (positif).

> **Justification clé pour le rapport** :
> En utilisant CIELab, le réseau prend directement le canal $L$ en entrée et n'a besoin de prédire que les canaux $(a, b)$. La tâche est simplifiée : le modèle n'a pas besoin de reconstruire la géométrie de la scène, il a pour unique rôle de "peindre" la chrominance.
> Pour reconstruire l'image finale, on concatène $L$ original avec les composantes $(\hat{a}, \hat{b})$ prédites, puis on applique la transformation inverse vers RGB.

### Normalisation
- $L \in [0, 100] \to$ normalisé dans $[-1, 1]$ via : $L_{norm} = \frac{L}{50} - 1$.
- $a, b \in [-128, 127] \to$ normalisés dans $[-1, 1]$ via : $a_{norm} = \frac{a}{128}$, $b_{norm} = \frac{b}{128}$.
- En sortie du réseau, une fonction d'activation **Tanh** borne naturellement les prédictions dans $[-1, 1]$.

---

## 3. Architecture U-Net & Rôle des Skip-Connections

Un simple réseau de classification réduit la taille de l'image (ex: $128 \times 128 \to 1 \times 1$) pour en extraire une étiquette. Or, pour la colorisation, nous devons générer une prédiction **pixel par pixel** à la même résolution que l'entrée ($128 \times 128$).

### Structure en "U"
1. **Encodeur (Partie descendante)** :
   - Enchaîne des convolutions et du MaxPooling ($128 \to 64 \to 32 \to 16$).
   - **Rôle** : Augmenter le champ réceptif des neurones. Plus on descend dans le réseau, plus les filtres analysent une large zone de l'image, ce qui permet de comprendre le **contexte sémantique global** (ex: "cette texture appartient à un pétale de fleur", "cette zone est de l'herbe").
2. **Goulet d'étranglement (Bottleneck)** :
   - C'est la couche la plus profonde (faible résolution, grand nombre de canaux). Elle capture les représentations abstraites de haut niveau.
3. **Décodeur (Partie montante)** :
   - Utilise des convolutions transposées (`ConvTranspose2d`) pour doubler la résolution à chaque étape ($16 \to 32 \to 64 \to 128$).
4. **Le rôle crucial des Skip-Connections** :
   - *Problème du simple encodeur-décodeur* : Les opérations de pooling détruisent l'information spatiale fine (les bordures nettes des pétales, les nervures des feuilles).
   - *Solution U-Net* : Chaque niveau du décodeur reçoit par concaténation la carte de caractéristiques exacte issue du niveau correspondant de l'encodeur.
   - **Bénéfice** : Le réseau combine à la fois la sémantique globale (décodeur) et les détails spatiaux haute résolution d'origine (skip-connections), évitant que la couleur ne déborde en dehors des contours.

---

## 4. Première Approche : U-Net + MSE (Baseline)

### Fonction de perte MSE (Mean Squared Error)
$$L_{MSE} = \frac{1}{N} \sum_{i=1}^N \left( (a_i - \hat{a}_i)^2 + (b_i - \hat{b}_i)^2 \right)$$

### Pourquoi les images produites sont-elles souvent ternes / brunâtres ?
Dans votre rapport, c'est **le point théorique le plus important à expliquer** :
- La perte MSE pénalise les grandes erreurs de façon quadratique.
- Lorsqu'une fleur peut être rouge vif ($(a, b) = (+80, +20)$) ou bleue vif ($(a, b) = (-20, -70)$) avec une probabilité égale, prédire le rouge alors que l'image était bleue donne une énorme erreur MSE.
- Pour minimiser l'erreur moyenne sur l'ensemble du dataset, la solution mathématique optimale pour la MSE est de prédire **l'espérance conditionnelle** :
  $$\hat{y}_{opt} = \mathbb{E}[(a, b) \mid L]$$
- Or, dans l'espace Lab, la moyenne de couleurs vives opposées converge vers le centre $(a=0, b=0)$, c'est-à-dire un **gris neutre / sépia** !
- Le modèle préfère donc prédire une couleur peu saturée et prudente plutôt que de risquer une couleur vive mais fausse.

---

## 5. Méthodologie d'Évaluation (Section 5 du Sujet)

Pourquoi la seule valeur de la MSE d'entraînement ne suffit pas ?
Un modèle qui prédirait toujours $(a=0, b=0)$ (une image en noir et blanc) obtiendrait une MSE relativement faible, tout en étant visuellement totalement inutile !

Pour évaluer rigoureusement le modèle, nous avons implémenté dans `evaluate.py` :
1. **$\Delta ab$ (Distance chromatique euclidienne)** :
   $$\Delta ab = \sqrt{(a - \hat{a})^2 + (b - \hat{b})^2}$$
   Mesurée dans l'espace Lab réel, elle correspond à la formule standard de différence de couleur (CIE76). Une valeur plus faible indique une meilleure fidélité globale aux couleurs d'origine.
2. **Dispersion chromatique ($\sigma_a, \sigma_b$)** :
   - On calcule l'écart-type des prédictions $\sigma(\hat{a}), \sigma(\hat{b})$ et on le compare à celui de la vérité terrain $\sigma(a_{vrai}), \sigma(b_{vrai})$.
   - **Interprétation** : Si $\sigma(\hat{a})$ est proche de 0, le modèle a "triché" en s'effondrant vers le gris. Si $\sigma(\hat{a})$ est élevé et proche de la réalité, le modèle produit des couleurs franches et variées.

---

## 6. Piste d'Amélioration : Transfert d'Apprentissage (ResNet-18)

### Motivation
- La baseline U-Net possède environ 2 millions de paramètres initialisés aléatoirement. Entraîné sur un dataset modeste (~6 500 images de train), l'encodeur doit apprendre péniblement à la fois à détecter les contours basiques et à comprendre la nature des objets.
- **Principe du Transfert d'Apprentissage** : On utilise le réseau **ResNet-18**, préalablement entraîné sur la gigantesque base de données **ImageNet** (plus de 1,2 million d'images). Ses couches convolutives constituent un extracteur de caractéristiques visuelles extrêmement puissant et robuste.

### Intégration dans notre architecture (`ResNetUNet`)
1. **Adaptation de la première couche** : ResNet attend 3 canaux RGB en entrée. Pour lui fournir notre canal de luminance $L$ unique, nous avons moyenné les poids de ses filtres initiaux sur les 3 canaux, préservant ainsi sa sensibilité aux textures sans rajouter de paramètres aléatoires.
2. **Gel de l'encodeur (`freeze_encoder = True`)** :
   - Lors de l'entraînement, les gradients ne modifient pas les poids de ResNet. Seul le décodeur apprend à traduire les caractéristiques extraites vers les couleurs $(a, b)$.
   - **Avantages** : Convergence beaucoup plus rapide, risque de surapprentissage (overfitting) fortement diminué, temps de calcul réduit sur le GPU.

---

## 7. Tableau Comparatif des Résultats (À remplir pour votre rapport)

Exécutez `python evaluate.py` sur les deux modèles pour compléter ce tableau :

| Modèle | Test MSE (normalisée) | Erreur $\Delta ab$ (Lab) | $\sigma(\hat{a})$ (Écart-type a) | $\sigma(\hat{b})$ (Écart-type b) | Observations visuelles |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Vérité Terrain** | *0.000* | *0.00* | *(ex: ~22.5)* | *(ex: ~28.3)* | Couleurs vives et naturelles |
| **U-Net Baseline (de zéro)** | *ex: 0.0142* | *ex: 14.8* | *ex: 8.2* | *ex: 11.5* | Couleurs ternes, tendance au sépia |
| **ResNet-18 Transfert** | *ex: 0.0115* | *ex: 11.2* | *ex: 14.1* | *ex: 17.8* | Couleurs plus vives, contours respectés |

*(Insérez dans cette section la figure `comparison_output.png` générée par `python demo.py --compare`)*.

---

## 8. Déclaration d'Utilisation de l'IA Générative (Obligatoire dans le Sujet)

*Exemple de paragraphe de transparence à insérer à la fin de votre rapport :*

> **Déclaration relative à l'usage de l'IA générative :**
> Conformément aux consignes du projet, ce travail a bénéficié de l'assistance d'un modèle d'IA générative (Google Antigravity IDE). L'IA a été utilisée pour :
> 1. Structurer l'architecture du code PyTorch et modulariser les scripts (`load.py`, `model_resnet.py`, `evaluate.py`, `demo.py`).
> 2. Déboguer l'alignement des dimensions tensorielles lors des opérations de concaténation des skip-connections.
> 3. Vulgariser et clarifier les concepts mathématiques sous-jacents (tels que la relation entre la perte MSE et l'espérance conditionnelle dans l'espace CIELab).
> L'ensemble du code, des résultats expérimentaux et des analyses a été compris, testé et vérifié par mes soins.
