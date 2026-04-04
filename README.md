# 🎰 Mega Millions Positional Vector Probability Engine

A quantitative prediction engine for Mega Millions lottery analysis, combining three algorithmic layers:

- **Layer 1 — Hot/Cold Frequency** (exponential decay weighting)
- **Layer 2 — Positional Vector Probability** (per-position distribution modeling)
- **Layer 3 — Mega Ball Frequency Weighting**

Plus a production **Multi-Armed Bandit (MAB) ensemble** engine using Thompson Sampling and Kelly Criterion for ticket portfolio management.

---

## 📁 File Overview

| File | Description |
|------|-------------|
| `engine_v2.py` | Three-layer positional vector engine (v2.0) — core algorithm |
| `mab_engine.py` | Multi-Armed Bandit engine with Thompson Sampling + Kelly Criterion |
| `predictor.py` | v4.0 ensemble predictor — orchestrates MAB + strategies |
| `evaluator.py` | Backtesting and ticket evaluation against historical draws |
| `analyzer.py` | Frequency and pattern analysis utilities |
| `scraper.py` | Mega Millions results scraper |
| `fetch_html.py` | HTML fetcher utility for scraping |
| `utils.py` | Shared utility functions |
| `megamillions_history.csv` | Historical draw data |
| `mab_state.json` | Persisted MAB arm state between runs |
| `predictions.json` | Saved prediction outputs |
| `ALGORITHM.md` | Full bilingual (EN/ZH) algorithm design document |

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the v2.0 positional engine
python engine_v2.py

# 4. Run the full MAB ensemble predictor
python predictor.py

# 5. Evaluate past predictions
python evaluator.py
```

---

## 🧮 Algorithm Summary

### Layer 1: Hot/Cold Frequency (α = 0.35)
Applies exponential decay to weight recent draws more heavily than older ones:

```
score_i = Σ (freq_i_in_window × decay^t)
```

### Layer 2: Positional Vector Probability (β = 0.45)
Builds a probability distribution for each sorted position (P1–P5) from historical data, then scores a ticket by multiplying positional likelihoods:

```
pos_score = Π P(n | position_k)  for k in 1..5
```

This is the core innovation — numbers are evaluated not just by raw frequency but by how often they appear **at that specific sorted position**.

### Layer 3: Mega Ball Weighting (γ = 0.20)
Frequency-based scoring on the Mega Ball (1–25) with recency decay.

### Composite Score
```
final_score = α × hot_cold + β × positional + γ × mega_ball
```

See `ALGORITHM.md` for full mathematical derivation, complexity analysis, and tuning guide.

---

## 📊 Generation Strategy

Tickets are generated using **soft-band weighted sampling**:
1. Divide 1–70 into bands based on historical positional distributions
2. Sample from each position's distribution (not purely random)
3. Score all candidates with the composite formula
4. Emit top-K tickets

---

## ⚙️ Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | 0.35 | Hot/cold frequency weight |
| `beta` | 0.45 | Positional vector weight |
| `gamma` | 0.20 | Mega Ball weight |
| `window` | 100 | Historical draws to analyze |
| `decay` | 0.95 | Recency decay factor per draw |

---

## 📈 Recent Results

```
T01 [A-hot     ] [14, 29, 32, 43, 46] MB:4  →  0+0
T02 [B-cold    ] [17, 19, 24, 36, 44] MB:15  →  0+1  ⭐ WON $2
T03 [B-cold    ] [2, 17, 26, 48, 49]  MB:9  →  0+0
T06 [C-random  ] [18, 31, 48, 63, 69] MB:9  →  2+0
T07 [D-hybrid  ] [2, 12, 31, 32, 59]  MB:15  →  1+1  ⭐ WON $4
```

---

## 🔬 Complexity

| Operation | Time Complexity |
|-----------|----------------|
| Positional distribution build | O(N × 5) |
| Candidate generation | O(C × 5) |
| Composite scoring | O(C) |
| Total pipeline | O(N + C) |

Where N = historical draws, C = candidate pool size.

---

## ⚠️ Disclaimer

This is a statistical analysis and educational tool. Lottery draws are random — no algorithm can guarantee wins. Play responsibly.

---

## 📄 License

[Apache License 2.0](LICENSE)
