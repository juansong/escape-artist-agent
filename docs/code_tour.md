# Escape Artist Agent - Code Tour

## Overall Flow (big picture)
1. **Environment** (`escape_artist/envs/*`) defines the gridworld and layout sampling.
2. **Algorithms** (`escape_artist/algos/*`) learn tabular Q[s,a].
3. **Scripts** (`experiments/*`) train, evaluate, and make figures/GIFs.
4. **Utils** (`escape_artist/utils/*`) draw curves, heatmaps, rollouts.
5. **Configs** (`configs/*.yaml`) parametrize env+algo so runs are reproducible.
6. **Artifacts** land in `runs/<name>/` (Q.npy, returns.npy, figs/) and curated `assets/` for README.

## `escape_artist/envs/generators.py`
Helpers for random map generation + solvability.

```python
EMPTY = 0
TRAP = 1
GOAL = 2
```

#### **`exclusion_mask(size, start, goal, r_safe) -> np.ndarray[bool]`**
- Purpose: Mark a Chebyshev square around start/goal where traps are disallowed.
- Args: `size=(H,W)`, `(start_x,start_y)`, `(goal_x,goal_y)`, `r_safe:int`.
- Returns: Boolean mask where True = do not place traps.

#### **`path_exists_bfs(grid, start, goal) -> bool`**
- Purpose: Ensures a path exists from start to goal avoiding traps.
- Details: 4-connected BFS over `grid != TRAP`.

#### **`sample_trap_layout(rng, size, start, goal, traps_pct, r_safe, max_resample=50) -> np.ndarray[int8]`**
Purpose: Generate a random layout that (a) respects exclusion zones, (b) is solvable.
Behavior: Samples traps by Bernoulli(`traps_pct`) outside exclusions, sets `GOAL`, resamples until BFS-solvable (or gives a trap-sparse fallback).
Returns: `grid` with values in `{EMPTY, TRAP, GOAL}`.

> Note: High `traps_pct` with small `r_safe` can increase resampling time.

---

## `escape_artist/envs/escape_env.py`
Gymnasium-style envrionment (no hard dependency on Gym).

#### **`@dataclass EnvConfig`**
Key fields:
- **Geometry**: `size(H,W)`, `start(x,y)`, `goal(x,y)`
- **Stochasticity**: `slip` (probability action is replaced by a random one)
- **Traps**: `traps_pct`, `r_safe`, `lethal_traps`
- **Rewards**: `step_cost`, `trap_penalty`, `goal_reward`
- **Rollout**: `max_steps`, `layout_mode={"per_env","per_episode"}`, `seed`, `obs_mode`

#### **`class EscapeEnv`**

- **Attributes**: `h`, `w`, `grid`, `agent_pos`, `action_space.n==4`, `_layout_id` (int)
- **Hidden helpers**:
    - `_validate()` — asserts config sanity (bounds, ranges)
    - `_generate_layout()` — calls `sample_trap_layout()`; sets `grid` & `_layout_id`
    - `_obs_from_pos(pos)` — returns observation (currently `[x,y]`)

#### **API**
- `reset(seed=None) -> (obs, info)`
    - Creates a fresh layout if `layout_mode="per_episode"` (or first call).
    - Resets agent to start.
    - `info` contains `pos`, `is_trap`, `is_goal`, `layout_id`.

- `step(action:int) -> (obs, reward:float, terminated:bool, truncated:bool, info)`
    - Clips movement to grid.
    - Applies slip; sets trap/goal flags.
    - **Termination**: goal or lethal trap; **Truncation**: time limit.
    - Reward = step_cost + (optional penalties/rewards).

---

## **`escape_artist/envs/algos/mc_control.py`**
On-policy Monte Carlo control (First-Visit / Every-Visit) with ε-soft behavior.

#### **`@dataclass MCConfig`**
- `episodes`, `gamma`
- `visit={"first","every"}`
- `epsilon_start`, `epsilon_end`, `epsilon_decay_episodes`
- `seed`

#### **`train_mc_control(env, cfg:MCConfig) -> (Q:np.ndarray[nS,4], returns:np.ndarray[episodes])`**
- Loop: For each episode, roll out with ε-soft policy, compute returns `G_t`, update `Q` by visit rule.
- Outputs: `Q.npy`-ready array; per-episode returns (for curves).

#### **`greedy_policy_rollout(env, Q, max_steps=None) -> (total_return, trajectory:list[(x,y)])`**
- Purpose: Execute the greedy policy w.r.t. `Q`; used for plots/eval.

---

## **`escape_artist/algos/mc_offpolicy.py`**