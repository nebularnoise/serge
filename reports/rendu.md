# Rendu du projet ML

Le rendu final à donner à Philippe + Adrien serait:

* Un dataset `pytorch` contenant les spectrogrammes annotés
* Un script Python permettant l'obtention d'un modèle `model.pt` de type **V(W)AE**, dont l'entrée / sortie est un spectrogramme d'amplitude en mel-frequency.
* Un script Python permettant à partir d'un modèle `model.pt` l'obtention d'un modèle génératif `generative.pt` prenant en entrée un point de l'espace latent (réduit ou pas), et donnant en sortie une waveforme. Ce modèle serait la concaténation de `model.pt` ainsi que du **MCNN**.
* Un external puredata `wae-sampler` servant de wrapper polyphonique pour le modèle `generative.pt`.
* Un patch puredata permettant le contrôle et la visualisation du parcours de l'espace latent.

Ça représente pas mal de boulot, et on a **2 semaines**, je vous propose donc deux choses: un planning de rendus à faire, ainsi que quelques prototypes des objets en question.

## puredata

L'external `wae-sampler` contient un `inlet` prenant une liste de 4 flottants, et un `outlet` signal. Lorsque qu'on **bang** un message contenant 4 flottants dans l'`inlet`, `wae-sampler` libère un de ses buffers et y écrit la waveforme correspondant à l'input, puis écrit dans le buffer de sortie la somme de tous les buffers (polyphonie).

Le choix des points dans l'espace latent se fera par le parcours de cercles dans un espace bidimensionnel. Pour un espace latent de 4 dimensions, on peut représenter un parcours par deux cercles. On peut voir ça un peu à la manière d'un step sequencer.

Dans le patch de contrôle, il y aura 10 paramètres en tout qui seront contrôlable. Ces paramètres sont:

* **x1**: origine du paramètre 1
* **y1**: origine du paramètre 2
* **x2**: origine du paramètre 3
* **y2**: origine du paramètre 4


* **d1**: taille du cercle 1 autour du point (x1,y1)
* **d2**: taille du cercle 2 autour du point (x2,y2)


* **r1**: randomness du choix du sample pour le cercle 1
* **r2**: randomness du choix du sample pour le cercle 2


* **s1**: nombre de pas pour faire le tour du cercle 1
* **s2**: nombre de pas pour faire le tour du cercle 2


Ca parait compliqué, mais avec un dessin ça passe mieux. Si ça vous va, je m'expliquerai un peu plus là dessus. Si ça vous tente, on pourra aussi faire de la visualisation avec la librairie **GEM** pour puredata, je vous montrerai c'est le feu.

## generative.pt

Le model `generative.pt` est la concaténation d'un **V(W)AE** avec un **MCNN**. Le **MCNN**, on ne s'en occupe pas. Le **V(W)AE** est basé sur ce que l'on a déjà fait. La seule méthode dont on a besoin pour l'exploitation en external est la méthode **forward**:

```C++
float *GenerativeModel::forward(float *point_espace_latent);
```

Cette méthode renvoie un nombre fixé de samples (à définir).

## planning

Je pense que ce serait bien qu'on définisse un planning de rendus à faire, histoire qu'on avance avant le projet de brigitte. Voici mes propositions:

* **dataset**, **generative.pt**, **wae-sampler**, **patch pd**: 8 janvier. A noter que les quatre trucs sont pseudos indépendants, en se basant sur un model bidon on peut faire l'external, tout comme on peut faire le model generative avec un dataset bidon. Je parle même pas du patch puredata.

* **Regroupement des trois modules**: 10 janvier. Si on se voit ça peut aller rapidement.

* **Entrainement du modèle sur les datasets**: 12 janvier. Normalement, ça devrait aller.

* **Finalisation du projet**: 14 janvier. En gros il s'agit de mettre au propre le code, commenter les fonctions, s'assurer de la compilation des externals, cleaner le repo...

**Il est également à noter qu'il faut continuer le compte rendu tout pendant ce temps là**.
