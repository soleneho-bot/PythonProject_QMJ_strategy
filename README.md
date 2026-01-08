# Enhanced Quality Minus Junk (QMJ) Strategy Implementation

Ce projet présente une implémentation avancée de la stratégie d'investissement quantitative "Quality Minus Junk", initialement théorisée par Asness, Frazzini et Pedersen (AQR Capital Management).

## Fondements Théoriques
La stratégie repose sur l'exploitation du facteur Qualité par la construction d'un portefeuille Long/Short. Elle identifie les entreprises présentant des caractéristiques de rentabilité élevée, de croissance stable et de risque financier réduit.

### Architecture des quatre piliers (AQR)
1. **Profitability** : Évaluation via le Gross Profits over Assets (GPOA).
2. **Growth** : Analyse de la dynamique de rentabilité sur une période de 5 ans.
3. **Safety** : Sélection basée sur un levier financier maîtrisé et un Beta de marché faible.
4. **Payout** : Analyse des politiques de distribution de dividendes et de rachats d'actions.

## Optimisation Dynamique du Modèle

Les données fondamentales (Bilan, Compte de résultat) sont publiées avec un décalage temporel et ne changent qu'au rythme des trimestres. Pour pallier ce problème, nous avons donc décidé d’intégrer d’autres indicateurs, dont la mise à jour est plus régulière / quotidienne :  

* **Exposant de Hurst (Analyse R/S)** : Application des théories de Mandelbrot (1963) pour mesurer la persistance des tendances et distinguer la croissance structurelle du bruit transactionnel.
* **Risque de Crédit Implicite** : Mesure de la corrélation avec l'ETF HYG (High Yield) pour isoler les composantes de dette risquée au sein de l'equity, conformément au modèle de Merton (1974).

## Architecture du Dépôt
* `/research` : Travaux de recherche. 
* `/src` : Code source de production pour l'analyse quotidienne et la gestion de base de données.
* `requirements.txt` : Spécifications des dépendances logicielles.

