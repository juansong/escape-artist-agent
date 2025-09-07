# CLI Walkthrough

> Run all commands from the repo root keeping the same Python interpreter used to install the package.

## 1) Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Sanity: same interpreter for everything

```bash
python - <<'PY'
import sys, escape_artist
print("Python:", sys.executable)
print("escape_artist:", escape_artist.__file__)
PY
```

Optional: quick test

```bash
python -m pytest -q
```
---

## 2) Train Easy baselines

MC First-Visit (easy)

```bash
python -m experiments.run_experiment --algo mc --visit first \
  --episodes 3000 --config configs/easy.yaml --out runs/mc_first_easy_ok
```

Q-Learning (easy)

```bash
python -m experiments.run_experiment --algo q \
  --episodes 3000 --config configs/easy.yaml --out runs/q_easy_ok
```
---

## 3) Evaluaate Easy (generates CSV + table)

```bash
python -m experiments.evaluate \
  --config configs/easy.yaml --episodes 200 \
  --runs runs/mc_first_easy_ok runs/q_easy_ok \
  --labels "MC (First)" "Q-Learning" \
  --out_csv assets/eval_easy.csv --out_md assets/eval_easy.md
```
---

## 4) Train Medium baselines

MC Every-Visit

```bash
python -m experiments.run_experiment --algo mc --visit every \
  --episodes 12000 --config configs/medium.yaml --out runs/mc_every_medium_long
```

Off-policy MC (Weighted IS)

```bash
python -m experiments.run_experiment --algo mc_off --is weighted \
  --episodes 18000 --config configs/medium.yaml --out runs/mc_off_weighted_medium_long
```

Q-Learning

```bash
python -m experiments.run_experiment --algo q \
  --episodes 12000 --config configs/medium.yaml --out runs/q_medium_long
```
---

## 5) Evaluate Medium

```bash
python -m experiments.evaluate \
  --config configs/medium.yaml --episodes 200 \
  --runs runs/mc_every_medium_long runs/mc_off_weighted_medium_long runs/q_medium_long \
  --labels "Every-Visit MC (12k)" "MC-OFF (Weighted, 18k)" "Q-Learning (12k)" \
  --out_csv assets/eval_medium.csv --out_md assets/eval_medium.md
```
---

## 6) Rebuild per-run plots

```bash
python -m experiments.run_experiment --plot --from runs/mc_every_medium_long
python -m experiments.run_experiment --plot --from runs/mc_off_weighted_medium_long
python -m experiments.run_experiment --plot --from runs/q_medium_long
```
---

## 7) Create README figures (assets/)

A) Combined learning curves (Medium)

```bash
python -m experiments.combine_curves \
  --runs runs/mc_every_medium_long runs/mc_off_weighted_medium_long runs/q_medium_long \
  --labels "MC (Every)" "MC-OFF (Weighted)" "Q-Learning" \
  --out assets/curve_medium_mc_mc-off_q.png
```


B) Hero GIF (3 frames: rollout from each run)

```bash
python -m experiments.make_hero_gif \
  --frames runs/mc_every_medium_long/figs/greedy_rollout.png \
          runs/mc_off_weighted_medium_long/figs/greedy_rollout.png \
          runs/q_medium_long/figs/greedy_rollout.png \
  --out assets/escape-artist-hero.gif \
  --fps 2
```

C) Layout montage (variety snapshot)

```bash
python -m experiments.make_layout_montage \
  --rows 3 --cols 4 --size 10 10 --traps_pct 0.10 --slip 0.1 \
  --out assets/layout_montage.png
```

D) Copy one clean overlay (heatmap + rollout) to assets/figs for README

```bash
mkdir -p assets/figs
cp runs/q_medium_long/figs/value_heatmap_policy.png assets/figs/value_heatmap_policy.png || true
cp runs/q_medium_long/figs/greedy_rollout.png      assets/figs/greedy_rollout.png      || true
```

## 8) (Optional) Ablations (small grid; fast)

```bash
python -m experiments.ablations \
  --size 8 8 --traps 0.05 0.15 --slips 0.0 0.1 \
  --episodes 600 --eval_eps 10 \
  --out_csv assets/ablations_quick.csv \
  --out_png_prefix assets/ablations_quick
  ```
  -----