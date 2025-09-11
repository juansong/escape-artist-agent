# Escape Artist Agent - Developer Guide

## Overview
1. **Environment** (`escape_artist/envs/*`) defines the gridworld and layout sampling.
2. **Algorithms** (`escape_artist/algos/*`) learn tabular Q[s,a].
3. **Scripts** (`experiments/*`) train, evaluate, and make figures/GIFs.
4. **Utils** (`escape_artist/utils/*`) draw curves, heatmaps, rollouts.
5. **Configs** (`configs/*.yaml`) parametrize env+algo so runs are reproducible.
6. **Artifacts** land in `runs/<name>/` (Q.npy, returns.npy, figs/) and curated `assets/` for README.
---

Full docstrings & additional notes.
---

## `escape_artist/envs/generators.py`
Map utilites (helpers for random map generation + solvability).

#### **`exclusion_mask(size, start, goal, r_safe) -> np.ndarray[bool]`**
- Purpose: Mark a Chebyshev square around start/goal where traps are disallowed.
- Args: 
    - `size=(H,W)` - `(H,W)` grid size.
    - `(start_x,start_y)` - `(x,y)` start.
    - `(goal_x,goal_y)` - `(x.y)` goal.
    - `r_safe:int` - Chebyshev radius where traps are disallowed around start/goal.
- Returns: 
    - `mask: np.ndarray[bool]` of shape `(H, W)`, `True` where traps are not allowed

#### **`path_exists_bfs(grid, start, goal) -> bool`**
- Purpose: Ensures a path exists from start to goal avoiding traps.
- Args:
    - `grid: np.ndarray[int]` — `(H, W)` with cell types `{EMPTY=0, TRAP=1, GOAL=2}`.
    - `start, goal: tuple[int,int]` — `(x, y)` positions.
- Returns:
    - `bool` - `True` if a 4-connected, trap-free path exists start -> goal.

> 4-connected BFS over `grid != TRAP`.

#### **`sample_trap_layout(rng, size, start, goal, traps_pct, r_safe, max_resample=50) -> np.ndarray[int8]`**
- Purpose: Generate a random layout that (a) respects exclusion zones, (b) is solvable.
- Behavior: Samples traps by Bernoulli(`traps_pct`) outside exclusions, sets `GOAL`, resamples until BFS-solvable (or gives a trap-sparse fallback).
- Args: 
    - `rng: np.random.Generator`
    - `size: (H, W)`, `start: (x, y)`, `goal: (x, y)`
    - `traps_pct: float` — Bernoulli prob for trap placement (outside exclusions).
    - `r_safe: int` — exclusion radius.
    - `max_resample: int` — retries until solvable.
- Returns:
    - `grid: np.ndarray[int8]` with `{EMPTY, TRAP, GOAL}`, solvable per BFS (or trap-sparse fallback after retries).

> Note: High `traps_pct` with small `r_safe` can increase resampling time.
---

## `escape_artist/envs/escape_env.py`
Gymnasium-style environment (no hard dependency on Gym).

#### **`@dataclass EnvConfig`**
Key fields:
- **Geometry**: `size(H,W)`, `start(x,y)`, `goal(x,y)`
- **Stochasticity**: `slip: float` - probability action is replaced by a random one
- **Traps**: `traps_pct: float`, `r_safe: int`, `lethal_traps: bool`
- **Rewards**: `step_cost`, `trap_penalty`, `goal_reward`
- **Episode**: `max_steps: int`, `layout_mode={"per_env","per_episode"}`, `seed: int|None`
- **Obs**: `obs_mode: str` (default `"pos"`)

#### **`class EscapeEnv(cfg: EnvConfig)`**
- Args(constructor):
  - `cfg: EnvConfig` - full environment configuration.
- Attributes: `h, w: int`, `grid: np.ndarray`, `agent_pos: (x,y)`, `action_space.n==4`, `_layout_id: int`
- Methods:
    - `_validate()` — internal assertions (sizes, bounds, ranges)
    - `_generate_layout()` — internal; creates `grid` via `sample_trap_layout`.
    - `_obs_from_pos(pos)` — returns observation `[x, y]` as `np.int32`.

#### **API**
Gymnasium API.
- `reset(seed: int|None = None) -> (obs: np.ndarray, info: dict)`
    - Creates a fresh layout if `layout_mode="per_episode"` (or first call).
    - Resets agent to start.
    - `info` contains `pos`, `is_trap`, `is_goal`, `layout_id`.
    - Args:
      - `seed: int|None` - optional RNG reseed for this reset.
    - Returns:
      - `info: dict` - `{"pos": (x,y), "is_trap": bool, "is_goal": bool, "layout_id": int}`.

- `step(action: int) -> (obs, reward:float, terminated:bool, truncated:bool, info)`
    - Clips movement to grid.
    - Applies slip; sets trap/goal flags.
    - **Termination**: goal or lethal trap; **Truncation**: time limit.
    - Args:
      - `action: int` - `0:up, 1:right, 2:down, 3:left` (with slip applied).
    - Returns:
      - `obs: np.ndarray[int32]` — `[x, y]` after move.
      - `reward: float` - step_cost + (optional penalties/rewards).
      - `terminated: bool` — reached goal or lethal trap.
      - `truncated: bool` — time limit exceeded.
      - `info: dict` — includes `pos`, `is_trap`, `is_goal`, `layout_id`.
---

## **`escape_artist/algos/mc_control.py`**
On-policy Monte Carlo control (First-Visit / Every-Visit) with ε-soft behavior.

#### **`@dataclass MCConfig`**
- `episodes: int`, `gamma: float`
- `visit: Literal["first","every"]`
- `epsilon_start: float|int`, `epsilon_end: float|int`, `epsilon_decay_episodes: float|int`
- `seed: int|None`

#### **`train_mc_control(env: EscapeEnv, cfg: MCConfig) -> (Q: np.ndarray, returns: np.ndarray)`**
- Loop: For each episode, roll out with ε-soft policy, compute returns `G_t`, update `Q` by visit rule.
- Outputs: `Q.npy`-ready array; per-episode returns (for curves).
- Args:
  - `env: EscapeEnv` - tabular grid env.
  - `cfg: MCConfig`
- Returns:
  - `Q: np.ndarray` - shape `(H*W, 4)`.
  - `returns: np.ndarray[episodes]` - episodic returns for plotting.

#### **`greedy_policy_rollout(env: EscapeEnv, Q: np.ndarray, max_steps: int|None = None) -> (G: float, traj: list[tuple[int,int]])`**
- Purpose: Execute the greedy policy w.r.t. `Q`; used for plots/eval.
- Args:
  - `env: EscapeEnv`
  - Q: np.ndarray - `(H*W, 4)`.
- Returns:
  - `G: float` - total discounted return along the trajectory.
  - `traj: list[(x,y)]` - visited positions step-by-step.
---

## **`escape_artist/algos/mc_offpolicy.py`**
Off-policy MC via importance sampling.

#### **`@dataclass OffMCConfig`**
- `episodes`, `gamma`
- `is_type: Literal["ordinary","weighted"]`
- Behavior ε: `epsilon_behavior_start/end/decay_episodes`
- `seed`

#### **`train_mc_offpolicy(env: EscapeEnv, cfg: OffMCConfig) -> (Q, returns)`**
- **Core**: Generate behavior episodes (ε-soft), compute target greedy returns with IS weights.
- **IS variants**: Ordinary vs Weighted (lower variance).
- **Output**: learned `Q`, episodic returns.
- Args:
  - `env: EscapeEnv`
  - `cfg: OffMCConfig`
- Returns:
  - `Q: np.ndarray (H*W,4)`
  - `returns: np.ndarray[episodes]`
---

## **`escape_artist/algos/q_learning.py`**
Tabular Q-Learning baseline.

#### **`@dataclass QLConfig`**
- `episodes`, `gamma`
- Learning rate schedule: `alpha_start/end/decay_episodes`
- Exploration schedule: `epsilon_start/end/decay_episodes`
- `seed`

#### **`train_q_learning(env: EscapeEnv, cfg: QLConfig) -> (Q, returns)`**
- Core: TD updates `Q[s,a] += α (r + γ max_a' Q[s',a'] − Q[s,a])`.
- Output: `Q`, per-episode returns.
- Args:
  - `env: EscapeEnv`
  - `cfg: QLConfig`
- Returns: 
 - `Q: np.ndarray (H*W,4)`
  - `returns: np.ndarray[episodes]`
---

## **`escape_artist/utils/plotting.py`**
Figure helpers (save to `.../figs/`). s

#### **`plot_learning_curve(returns, out_dir, title="Learning Curve") -> pathlib.Path`**
- Saves `learning_curve.png` under `<out_dir>/figs/`.
- Args:
  - `returns: np.ndarray` — per-episode returns.
  - `out_dir: Pathlike` — run or assets directory.
  - `title: str`
- Returns:
  - `path: Path` to `figs/learning_curve.png`.

#### **`compute_V_and_policy(Q: np.ndarray) -> (V: np.ndarray, greedy: np.ndarray)`**
- Args:
  - `Q: np.ndarray (H*W,4)`
- Returns:
  - `V: np.ndarray (H*W,4)` - `max_a Q[s,a]`.
  - `greedy: np.ndarray (H*W,)` - `argmax_a Q[s,a]`

#### **`plot_value_heatmap_and_policy(env, Q, out_dir, title="...") -> pathlib.Path`**
- Saves `value_heatmap_policy.png` showing `V(s)` heatmap and greedy arrows.
- Args:
  - `env: EscapeEnv`, `Q: np.ndarray`, `out_dir: Pathlike`, `title: str`
- Returns:
  - `path: Path` to `figs/value_heatmap_policy.png`.

#### **`plot_greedy_rollout(env, Q, greedy_policy_rollout, out_dir, max_steps) -> pathlib.Path`**
- Runs a single greedy rollout and overlays the trajectory; saves `greedy_rollout.png`.
- Args:
  - `env: EscapeEnv`, `Q: np.ndarray`
  - `greedy_policy_rollout: callable` — function from `mc_control`.
  - `out_dir: Pathlike`
  - `max_steps: int`
- Returns:
  - `path: Path` to `figs/greedy_rollout.png`.

> All images land in <out_dir>/figs/.
---

## **`experiments/run_experiment.py` (CLI)**
Train or plot-only (single run directory).
- Args:
  - `--algo {mc|mc_off|q}`
  - `--visit {first|every}` (mc only)
  - `--is {weighted|ordinary}` (mc_off only)
  - `--episodes INT`, `--seed INT`
  - `--config PATH` — YAML with `env:` (and optional `algo:` defaults)
  - `--out DIR` — run folder to write artifacts to
  - `--plot` — plot-only mode
  - `--from DIR` — run folder to read artifacts from (for `--plot`)
- Returns / Artifacts:
  - In `--out`: `Q.npy`, `returns.npy`, `params.json`, `config_used.yaml`, `figs/{learning_curve,value_heatmap_policy,greedy_rollout}.png`.
  - Console logs (progress).

**Train mode (examples)**:

```bash
python -m experiments.run_experiment --algo mc --visit every \
  --episodes 8000 --config configs/medium.yaml --out runs/mc_every_medium
```

**Plot-only**:

```bash
python -m experiments.run_experiment --plot --from runs/q_medium
```
---

## **`experiments/evaluate.py` (CLI)**
Evaluate greedy policies from trained runs → CSV + Markdown table.
- Args:
  - `--config PATH` — env YAML (evaluation setting; typically `layout_mode=per_episode`)
  - --runs DIR ...` — one or more run folders (must contain `Q.npy`)
  - `--labels STR ...` — labels matching `--runs`
  - `--episodes INT` — number of eval episodes per method
  - `--seed INT` — RNG for tie-breaking
  - `--out_csv PATH` — CSV output (metrics per method)
  - `--out_md PATH` — Markdown table (paste into README)
- Returns / Artifacts:
  - `assets/eval_*.csv`, `assets/eval_*.md`
  - Console summary lines per method.
- Metrics (per method):
  - `success_rate`, `avg_steps_success`, `avg_steps_all`, `detection_rate`, `timeout_rate`.

```bash
python -m experiments.evaluate \
  --config configs/medium.yaml --episodes 200 \
  --runs runs/mc_every_medium runs/mc_off_weighted_medium runs/q_medium \
  --labels "Every-Visit MC" "MC-OFF (Weighted)" "Q-Learning" \
  --out_csv assets/eval_medium.csv --out_md assets/eval_medium.md
```

**Internals**
- Builds **env** from YAML (uses `layout_mode="per_episode"` for robustness).
- For each run, loads `Q.npy`, runs N evaluation episodes with greedy actions.
- Metrics: `success_rate`, `avg_steps_success`, `avg_steps_all`, `detection_rate`, `timeout_rate`.
- Saves a compact markdown table you can paste into README.
---

## **`experiments/combine_curves.py` (CLI)**
Merge multiple `returns.npy` series (learning curves) into a single PNG.
- Args:
  - `--runs DIR ...` — each must contain `returns.npy`
  - `--labels STR ...` — one label per run
  - `--out PATH` — output PNG (e.g., `assets/curve_*.png`)
- Returns / Artifacts:
  - Combined curves saved to `--out`.

```bash
python -m experiments.combine_curves \
  --runs runs/mc_every_medium runs/mc_off_weighted_medium runs/q_medium \
  --labels "MC (Every)" "MC-OFF (Weighted)" "Q-Learning" \
  --out assets/curve_medium_mc_mc-off_q.png
```
---

## **`experiments/make_hero_gif.py` (CLI)**
Animate a rollout from `Q.npy`.
- Args:
  - `--run DIR` — run folder containing Q.npy
  - `--config PATH` — env YAML
  - `--want {any|success|failure}` — search outcome
  - `--layout {per_env|per_episode}` — layout mode during search
  - `--seed INT`, `--max_tries INT`, `--fps INT`, `--hold_end INT`
  - `--out PATH` — output GIF path
- Artifacts:
  - Animated rollout path (successful/failed) GIF at `--out`.

```bash
python -m experiments.make_hero_gif \
  --run runs/q_medium_long --config configs/medium.yaml \
  --want success --layout per_env --out assets/rollout_success.gif --fps 4
```

> Tries seeds until it finds `success` (or `failure`). <br>
> Renders a frame per step; appends a short pause at the end.
---

## **`expeeriments/make_layout_montage.py` (CLI)**
Generate a tiled montage of random layouts.
- Args:
  - `--rows INT`, `--cols INT`
  - `--size H W`
  - `--traps_pct FLOAT`
  - `--slip FLOAT`
  - `--out PATH`
- Returns / Artifacts:
  - Montage PNG at `--out`.

```bash
python -m experiments.make_layout_montage \
  --rows 3 --cols 4 --size 10 10 --traps_pct 0.10 --slip 0.1 \
  --out assets/layout_montage.png
```
---

## **`experiments/ablations.py` (CLI)**
Ablation study sweeping over (traps_pct x slip) for ME(Every) and Q-Learning.
- Args:
  - `--size H W`
  - `--traps FLOAT ...`
  - `--slips FLOAT ...`
  - `--episodes INT` — training episodes per setting
  - `--eval_eps INT` — greedy eval episodes per setting
  - `--out_csv PATH`
  - `--out_png_prefix PATH` — prefix for saved heatmaps
  - `--seed INT`
- Returns / Artifacts:
  - CSV with grid of results.
  - Heatmaps like `<prefix>_mc.png`, `<prefix>_q.png`.
---

## **`experiments/analysis.ipynb` (Notebook)**
Notebook features:
- Combine learning curves (from run folders).
- Render value heatmap + greedy arrows and a greedy rollout on a fixed layout.
- Preview ablation CSVs → heatmaps.
- Build animated success/failure rollout GIFs via make_hero_gif.py.
- Render greedy rollout policy behavior by algorithm on a fixed layout.

Inputs:
  - Existing run folders with `Q.npy` and `returns.npy` (e.g., `runs/mc_every_medium`, `runs/q_medium`).
  - `configs/*.yaml`.

Outputs:
  - `assets/curve_*.png` (combined learning curves)
  - `assets/figs/value_heatmap_policy.png`
  - `assets/figs/greedy_rollout.png`
  - `assets/rollout_success.gif`, `assets/rollout_failed.gif` (via `make_hero_gif.py`)
  - 'assets/compare_mc_every.gif`, `assets/compare_mc_off.gif`, `assets/compare_q_learning.gif`
  - `assets/ablations_*.png` and ablation CSV preview

> First cell bootstraps %pip install -e .[dev] into the kernel and normalizes CWD. <br>
> Notebook files are ignored when calculating repository language statistics (`*.ipynb linquist-vendored`)
---

## **`tests/*`**
Pytest suites.
- `test_generators.py` — exclusions, BFS path, sampling solvability.
- `test_env.py` — reset/step bounds, slip effect, termination/truncation, `layout_id` behavior.
- `test_mc.py`, `test_q_learning.py` — "easy" learning sanity.

> Run `pytest -q` after `pip install -e .[dev]` for a "quick" test.
---

## **`configs/*.yaml`**
Readable presets for environment and default algo knobs.

**Keys**:
- `env:` - fields mirrored in `EnvConfig`.
- `algo:` - optional defaults for per-algo schedules.

**Common fields (under `env:`)**:
- `size`, `start`, `goal`, `traps_pct`, `r_safe`, `slip`, `lethal_traps`,
- `step_cost`, `trap_penalty`, `goal_reward`, `max_steps`,
- `layout_mode: per_env|per_episode`, `seed`, `obs_mode`.
---

## **`assets/`**
Curated, committed outputs for README:
- `escape-artist-hero.gif` / `rollout_success.gif` / `rollout_failed.gif`
- `curve_*.png`, `layout_montage.png`
- `eval_*.md`, `eval_*.csv`
- `figs/value_heatmap_policy.png`, `figs/greedy_rollout.png`

> Ensure `.gitignore` does not exclude `assets/**`.
---

## **`runs/`**
Per-run artifacts:
- `Q.npy`, `returns.npy`, `params.json`, `config_used.yaml`
- `figs/learning_curve.png`, `figs/value_heatmap_policy.png`, `figs/greedy_rollout.png`

> Files are gitignored.
---

## **`Makefile`**
QoL shortcuts:
- `make install`
- `make train-medium` (MC-Every, MC-OFF Weighted, Q on medium)
- `make figures` (combined curve, hero GIF, montage)
- `make clean-runs`, `make clean-assets`
---

## **Packaging / metadata**
- `pyproject.toml`- modern packaging, extras `[dev]` for test/figures.
- `requirements.txt` — optional pinned versions for lockstep installs.
- `.github/workflows/ci.yml` — optional CI running `pytest` on push.
- `LICENSE` — GNU
---

## **Conventions & gotchas**
- State indexing: `s = y * W + x` (row-major). `Q.shape == (H*W, 4)`.
- Observation: default `"pos"` → `np.array([x,y], dtype=int32)`.
- Success check: `info["is_goal"]` or `env.grid[y,x] == GOAL` (evaluation uses both).
- Per-episode vs per-env: training on `per_episode` stresses generalization; figures often use `per_env` for repeatable visuals.
- Seeds: env layout seeds and algo RNG seeds are separate; set both for reproducibility.
-----