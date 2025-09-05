# -------- Escape Artist Agent - Makefile --------
# Quick commands:
#   make install           # install package + dev deps
#   make train-medium      # run MC, MC-OFF, Q on medium config
#   make figures           # build comparison curve, hero GIF, and layout montage
#   make clean-runs        # remove runs/ artifacts

PY ?= python
CONFIG_DIR := configs
RUNS_DIR   := runs
ASSETS_DIR := assets

# Medium runs used in README figures
RUN_MC_MED     := $(RUNS_DIR)/mc_every_medium
RUN_MCOFF_MED  := $(RUNS_DIR)/mc_off_weighted_medium
RUN_Q_MED      := $(RUNS_DIR)/q_medium

# Figure outputs
CURVE_PNG := $(ASSETS_DIR)/curve_medium_mc_mc-off_q.png
HERO_GIF  := $(ASSETS_DIR)/escape-artist-hero.gif
MONTAGE_PNG := $(ASSETS_DIR)/layout_montage.png

.PHONY: install train-easy train-medium train-hard train-mc-medium train-mcoff-medium train-q-medium plots-medium figures fig-curve fig-hero-gif fig-montage ensure-assets clean-runs clean-assets clean

# ---------- Setup ----------
install:
	$(PY) -m pip install -e .[dev]

# ---------- Training (easy/medium/hard shortcuts) ----------
train-easy:
	$(PY) -m experiments.run_experiment --algo mc --visit first \
		--episodes 3000 --config $(CONFIG_DIR)/easy.yaml --out $(RUNS_DIR)/mc_first_easy

train-mc-medium:  ## MC (Every-Visit) on medium
	$(PY) -m experiments.run_experiment --algo mc --visit every \
		--episodes 8000 --config $(CONFIG_DIR)/medium.yaml --out $(RUNS_DIR)/mc_every_medium

train-mcoff-medium: ## MC-OFF (Weighted IS) on medium
	$(PY) -m experiments.run_experiment --algo mc_off --is weighted \
		--episodes 12000 --config $(CONFIG_DIR)/medium.yaml --out $(RUNS_DIR)/mc_off_weighted_medium

train-q-medium: ## Q-Learning on medium
	$(PY) -m experiments.run_experiment --algo q \
		--episodes 8000 --config $(CONFIG_DIR)/medium.yaml --out $(RUNS_DIR)/q_medium

train-medium: train-mc-medium train-mcoff-medium train-q-medium

train-hard:
	$(PY) -m experiments.run_experiment --algo mc_off --is ordinary \
		--episodes 15000 --config $(CONFIG_DIR)/hard.yaml --out $(RUNS_DIR)/mc_off_ordinary_hard

# ---------- Plot (re-generate figures for existing runs) ----------
plots-medium:
	$(PY) -m experiments.run_experiment --plot --from $(RUN_MC_MED)
	$(PY) -m experiments.run_experiment --plot --from $(RUN_MCOFF_MED)
	$(PY) -m experiments.run_experiment --plot --from $(RUN_Q_MED)

# ---------- README figures ----------
figures: ensure-assets fig-curve fig-hero-gif fig-montage

fig-curve: ensure-assets
	$(PY) -m experiments.combine_curves \
		--runs $(RUN_MC_MED) $(RUN_MCOFF_MED) $(RUN_Q_MED) \
		--labels "MC (Every)" "MC-OFF (Weighted)" "Q-Learning" \
		--out $(CURVE_PNG)

fig-hero-gif: ensure-assets
	$(PY) -m experiments.make_hero_gif \
		--frames $(RUN_MC_MED)/figs/greedy_rollout.png \
		         $(RUN_MCOFF_MED)/figs/greedy_rollout.png \
		         $(RUN_Q_MED)/figs/greedy_rollout.png \
		--out $(HERO_GIF) --fps 2

fig-montage: ensure-assets
	$(PY) -m experiments.make_layout_montage \
		--rows 3 --cols 4 --size 10 10 --traps_pct 0.10 --slip 0.1 \
		--out $(MONTAGE_PNG)

ensure-assets:
	mkdir -p $(ASSETS_DIR)

# ---------- Cleaning ----------
clean-runs:
	rm -rf $(RUNS_DIR)

clean-assets:
	rm -f $(CURVE_PNG) $(HERO_GIF) $(MONTAGE_PNG)

clean: clean-runs clean-assets
