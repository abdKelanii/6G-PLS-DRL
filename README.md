# 6G Physical Layer Security via Deep Reinforcement Learning

A research project that applies Deep Q-Network (DQN) reinforcement learning to mitigate physical-layer security threats in 6G wireless communication systems.

## Overview

This project models a 6G communication environment with a Base Station, a legitimate user, an eavesdropper (Eve), and a jammer. A DQN agent learns adaptive defense strategies — including power control, beamforming, and artificial noise injection — to maximize the secrecy rate against adversarial threats.

The agent is compared against two baselines:
- **Static Random Policy** — random action selection
- **Threshold-Based Heuristic Policy (TBHP)** — rule-driven action selection

| Policy | Avg. Reward |
|--------|-------------|
| DQN (Ours) | **4.09** |
| TBHP | 2.89 |
| Static Random | 2.51 |

DQN outperforms the static baseline by **62.93%** and the heuristic baseline by **41.39%**.

## Repository Structure

```
.
├── code/
│   └── main.ipynb                  # DQN training, evaluation, and visualizations
├── latex/
│   ├── conference_101719.tex       # IEEE conference paper (LaTeX source)
│   └── *.png                       # Figures used in the paper
├── paper/
│   └── paper.pdf                   # Submitted paper
├── citation-papers/                # Reference papers on 6G PLS and DRL
├── presentation/                   # Slide deck
└── website/                        # Astro-based project website
```

## Environment

The custom Gymnasium environment (`Secure6GEnv`) has:

- **State space** (4 continuous values): signal strength, jammer power, noise level, Eve distance
- **Action space** (5 discrete actions):
  - `0` — increase transmit power
  - `1` — decrease transmit power
  - `2` — beam left
  - `3` — beam right
  - `4` — inject artificial noise

The reward function maximizes the secrecy rate while penalizing jammer power and noise:

```
reward = secrecy_rate × 5 − jammer_power − noise_level × 0.5
```

## Getting Started

### Prerequisites

```bash
pip install stable-baselines3 gymnasium matplotlib numpy
```

### Run

Open and run `code/main.ipynb` cell by cell:

1. **Cells 1–2** — Install dependencies and import libraries
2. **Cell 3** — Define the `Secure6GEnv` Gymnasium environment
3. **Cells 4–5** — Train the DQN agent for 5,000 steps
4. **Cells 6–8** — Plot reward curves and run baseline comparisons
5. **Cells 9–10** — Print average rewards and compute improvement percentages
6. **Cells 11–12** — Track and plot secrecy rate over time
7. **Cells 13–14** — Visualize the static and dynamic network topology

## Key Results

- DQN learns to adaptively increase signal strength, steer beams, and inject noise in response to dynamic jammer and eavesdropper behavior.
- The secrecy rate stabilizes significantly higher than either baseline after training.

## References

See `citation-papers/` for the six papers this work builds on, covering GAN-powered DRL for IoT security, STAR-RIS-assisted multi-user ISAC, RL-based cross-layer security for 6G, covert communication with relay/RIS optimization, and advanced 6G security frameworks.
