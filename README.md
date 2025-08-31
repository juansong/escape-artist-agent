# 🎮 Monte Carlo Stealth Agent  

An implementation of **Monte Calro control** for a custom escape tactics game environment.
The agent learns to escape a grid world, avoid guards, and reach the extraction point through trial and error.

This project demonstrates how **reinforcement learning (RL)** - specifically **on-policy and off-policy Monte Carlo methods** - can be applied to
**game AI design**

---

## 🚀 Features  
- ✅ On-policy **First-Visit** and **Every-Visit Monte Carlo Control**  
- ✅ Off-policy Monte Carlo with **Importance Sampling**  
- ✅ Custom **grid-based escape environment** with guards, traps, and goals  
- ✅ Visualizations: Q(s,a) heatmaps, trajectory overlays, episodic return plots  
- ✅ Comparisons with **Q-learning** for benchmarking  

---

## 🛠️ Installation  

Clone the repo and install dependencies:  

```bash
git clone https://github.com/your-username/mc-stealth-agent.git
cd mc-stealth-agent
pip install -r requirements.txt
```

## ▶️ Usage

### Training the Agent  

Run Monte Carlo training:  

```bash
python experiments/train_mc.py --episodes 5000
```

Adjust hyperparameters in `experiments/config.yaml`.

### Evaluating Policies

```bash
python experiments/evaluate.py --model saved_models/mc_policy.pkl
```

This will run the agent with the trained policy and log metrics such as:

- Success rate
- Average steps per episode
- Detection rate
---

## 📊 Analysis & Visualization  

The `notebooks/` folder contains tools to:  
- Plot **episodic return distributions**  
- Generate **Q(s,a) heatmaps**  
- Overlay **learned trajectories** vs scripted baselines  

Example heatmap visualization:  

<!-- <p align="center">  
  <img src="docs/q_heatmap.png" width="500"/>  
</p>   -->
---

## 🎥 Demo  

<!-- <p align="center">  
  <img src="docs/demo.gif" width="500"/>  
</p>   -->

The demo shows the agent’s progression:  
- **Episode 1:** Random policy, frequent guard detection  
- **Episode 500:** Learns safer detours and risk avoidance  
- **Episode 5000:** Consistently reaches the goal with minimal steps  
---

## 📈 Results  

| Method                  | Success Rate ↑ | Avg Steps ↓ | Detection Rate ↓ |  
|--------------------------|----------------|-------------|------------------|  
| First-Visit MC           | 72%            | 18.4        | 12%              |  
| Every-Visit MC           | 76%            | 17.9        | 10%              |  
| Off-Policy MC (IS)       | 80%            | 16.7        | 9%               |  
| Q-Learning (baseline)    | 69%            | 19.5        | 15%              |  

**Key insights:**  
- Monte Carlo control learns safe navigation paths but requires many episodes.  
- Off-policy MC with importance sampling leverages scripted/human play data for faster convergence.  
- Reward shaping significantly influences the agent’s stealth style (riskier but faster vs. safer but slower).  

---


## 📂 Project Structure  

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```

--------
