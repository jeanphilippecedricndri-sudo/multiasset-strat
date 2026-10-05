# multiasset-strat

Stratégie systématique multi-asset construite sur [`portopt`](https://github.com/jeanphilippecedricndri-sudo/portfolio-opt) :
signaux prix, sélection de 10 lignes, budget de risque par poche, contraintes de turnover, backtest
walk-forward avec frais fixes par ordre, rapport et poids cibles de la prochaine période.

## Installation

```bash
pip install -e ".[dev]"      # installe portopt depuis GitHub
pytest                       # 11 tests, données simulées
```

## Lancement

**Backtest et recherche**

```bash
python -m strat.run                      # backtest sur données Yahoo depuis 2008 + rapport
python -m strat.run --synthetic          # hors ligne, données simulées
python -m strat.run --hold 5             # variante hebdomadaire
python -m strat.run --max-stock-weight 0.10
python -m strat.research                 # test de chaque alpha, par poche, train/test
```

**Initialisation du portefeuille (une seule fois)**

```bash
python -m strat.run --init --nav 10000           # aperçu des poids initiaux, rien n'est enregistré
python -m strat.run --init --nav 10000 --save    # le jour des ordres : crée portfolio/holdings.csv
```

**Rebalancement mensuel (toutes les 21 séances, soit environ une fois par mois)**

1. Mets à jour `portfolio/holdings.csv` avec la valeur réelle de chaque ligne chez ton courtier
   (colonne `value` en euros, tickers proxys). Le code normalise en poids.
   ```csv
   asset,value
   SHY,2141
   USO,2210
   CIBR,1702
   ```
2. Lance le calcul avec la valeur totale du jour. Le fichier est rechargé automatiquement.
   ```bash
   python -m strat.run --nav 10250
   ```
3. Lis la table « Poids cibles » : `HOLD` = ne rien faire, `BUY` / `SELL` avec le montant en euros
   dans `order_eur`. Les règles de turnover sont déjà appliquées (2 remplacements au plus, pas
   d'ordre sous 2 points d'écart).
4. Passe les ordres, puis enregistre le rebalancement :
   ```bash
   python -m strat.run --nav 10250 --save
   ```
   `portfolio/holdings.csv` contient alors les nouvelles cibles, et le détail est archivé dans
   `portfolio/history/rebalancing_AAAA-MM-JJ.csv`.

À chaque lancement, la console indique la date du dernier rebalancement enregistré et le nombre de
séances écoulées, avec un avertissement s'il y en a moins de 21.

Chaque lancement écrit dans `outputs/` :

| Fichier | Contenu |
|---|---|
| `report.html` | poids cibles, performance, contrôles de risque, IC du signal, graphiques |
| `next_weights.csv` | poids cibles de la prochaine période, écart au portefeuille détenu, ordres en EUR |
| `summary.csv`, `risk_checks.csv`, `worst_losses.csv`, `ic.csv`, `weights_history.csv` | tables du rapport |
| `charts/*.png` | graphiques `portopt.plot` + décomposition de variance + ordres |

### Poids de la prochaine période et suivi du portefeuille

Ce ne sont pas des prévisions de rendement : ce sont les poids que la stratégie fixe avec toutes
les données disponibles, à appliquer à la séance suivante. Chaque ligne porte le nom de l'actif.

Point de départ, par ordre de priorité : `--init` (cash), `--holdings fichier.csv`, sinon
`portfolio/holdings.csv` s'il existe, sinon le portefeuille simulé du backtest.

`portfolio/holdings.csv` n'est écrit qu'avec `--save`, et chaque rebalancement enregistré est
archivé dans `portfolio/history/rebalancing_AAAA-MM-JJ.csv`. Le fichier contient les poids
**cibles**. Avant le rebalancement suivant, remplace-les par la valeur réelle de chaque ligne
(colonne `value` en euros, tickers proxys) : le code normalise.

## Méthode

1. **Univers** (`universe.py`) : ~130 proxies US, 4 poches + cash (`SHY`).
2. **Signaux** (`alphas.py`, `signals.py`) : 13 alphas prix tirés de Velu et al. (ch. 5), *Finding
   Alphas* (ch. 5, 12, 18, 21, 29) et Isichenko (sec. 2.2) : momentum 12-1 et 6-1, TSMOM ajusté
   de la volatilité, ratios de moyennes mobiles, breakout, momentum résiduel, reversals 5 j / 21 j /
   résiduel, Bollinger, low vol, persistance. Chaque alpha est winsorisé et z-scoré par poche.
   Score v2 (`config.SIGNALS_V2`), choisi sur 2009-2017 puis vérifié une fois sur 2018-2026 :
   actions, ETF et commodities sur le composite de tendance
   (`ma_cross_50_200`, `ma_ratio_200`, `mom_6_1`, `mom_12_1`), obligations sur `bollinger_20`.
   Filtre absolu $P_t > \text{SMA}_{200}$ conservé. `python -m strat.research` refait les tests
   alpha par alpha (IC de rang, excès du top-n, split train/test, correction de Benjamini-Hochberg).
3. **Sélection** (`estimators.py`) : 5 actions, 2 ETF, 2 obligations, 1 commodity. Une ligne détenue
   ne sort que si elle casse sa tendance ou quitte le top 40 % de sa poche. Au plus 2 sorties par
   rebalancement. Un slot sans candidat est placé sur `SHY`.
4. **Allocation** : `portopt.RiskParity(budgets=...)` avec 40 % du risque aux actions, 24 % ETF,
   24 % obligations, 12 % commodities, covariance Ledoit-Wolf. Bornes de poids 5 % à 20 %
   (3 % minimum pour une action). Les bornes déforment les budgets : le budget actions est
   ré-imposé comme plafond par bissection.
5. **Turnover** : bande de non-transaction de 2 points, plafond de 40 % par rebalancement. Les
   bornes de poids priment sur le plafond, et le plafond est souple pour la dernière rotation.
6. **Backtest** (`backtest.py`) : `TurnoverBacktester` hérite de `portopt.Backtester`, transmet
   les poids détenus à l'estimateur et facture
   $c_t = \text{spread} \cdot \tau_t + f \cdot n^{\text{ordres}}_t / \text{NAV}_t$.
7. **Contrôles** (`risk.py`) : contribution au risque par poche, décomposition
   $\sigma_p^2 = \beta_p^2\sigma_m^2 + \sum_i w_i^2\sigma_{\varepsilon,i}^2 + \text{autres facteurs}$,
   pires pertes journalières par ligne. **Signal** (`evaluate.py`) : IC de rang par poche.

## Structure du code

Le package `strat/` compte 13 fichiers, listés dans l'ordre où la stratégie les traverse.

**Le pipeline**

| Fichier | Rôle |
|---|---|
| `config.py` | Tous les paramètres : fenêtre de 252 jours, cycle de 21 séances, nombre de lignes par poche, budgets de risque, bornes de poids, règles de turnover, frais. Contient aussi `SIGNALS_V2`, le choix des signaux par poche. C'est le fichier à modifier pour régler la stratégie. |
| `universe.py` | Les 132 tickers par poche, leurs noms (`NAMES`) et la table des équivalents UCITS. |
| `data.py` | Téléchargement des prix Yahoo via `portopt`, avec repli ticker par ticker si un lot échoue. Générateur de données simulées. |
| `alphas.py` | Bibliothèque des 13 signaux prix, une fonction par signal, avec sa référence dans les livres. |
| `signals.py` | Assemble les signaux retenus en un score par actif : winsorisation, z-score par poche, moyenne, plus le filtre $P_t > \text{SMA}_{200}$. |
| `estimators.py` | Le cœur de la stratégie. `SignalEstimator` enchaîne sélection (hystérésis, 2 sorties maximum), budget de risque via `portopt.RiskParity`, bornes de poids, puis règles de turnover. Contient aussi les estimateurs des benchmarks. |
| `backtest.py` | `TurnoverBacktester`, qui étend le `Backtester` de `portopt` : transmet le portefeuille détenu à l'estimateur et facture les frais fixes par ordre. |

**L'analyse**

| Fichier | Rôle |
|---|---|
| `risk.py` | Contrôles de risque : contribution au risque par poche, décomposition de la variance (marché, spécifique, autres facteurs), pires pertes journalières par ligne. |
| `evaluate.py` | Qualité du score composite utilisé par la stratégie : IC de rang par poche, affiché dans le rapport. |
| `research.py` | Test de chaque signal un par un, en amont, pour choisir lesquels retenir : split 2009-2017 / 2018-2026 et correction de Benjamini-Hochberg. Se lance à part. |

**Les sorties**

| Fichier | Rôle |
|---|---|
| `forecast.py` | Poids cibles de la prochaine période à partir du portefeuille détenu, avec l'écart et le montant des ordres. |
| `report.py` | Graphiques (via `portopt.plot`) et rapport HTML. |
| `run.py` | Point d'entrée : lit les options (`--init`, `--nav`, `--save`), enchaîne tout le reste et écrit `outputs/`. |

`__init__.py` expose les objets principaux, et `tests/test_strat.py` contient 11 tests (pas de
look-ahead, somme des poids à 1, bornes respectées, rotations plafonnées, comptabilité des frais,
formules des signaux).

**Pour modifier la stratégie**

- Régler un paramètre : `config.py`.
- Ajouter un signal : une fonction dans `alphas.py`, puis son nom dans `SIGNALS_V2`.
- Ajouter ou retirer un actif : `universe.py` (ticker, poche et nom).

## Limites connues

- **Biais de survie** : la liste d'actions est celle des grandes capitalisations d'aujourd'hui.
  Les performances passées de la poche actions sont surestimées.
- **Proxies US** : non achetables par un particulier en Europe. `universe.UCITS` donne des
  équivalents, à vérifier chez ton courtier ; les actions se traitent directement.
- **Devise** : backtest en USD, sans effet de change EUR/USD.
- **Benchmarks** : rebalancés seulement quand une ligne dérive de plus de 5 points.
