# 6G Physical Layer Security via Deep Reinforcement Learning

A research project — *"Mitigating Physical Layer Vulnerabilities in 6G Communications using Deep Reinforcement Learning"* — that benchmarks continuous-control (DDPG, PPO) and discrete-control (DQN, tabular Q-Learning) deep RL against a static random policy and a threshold-based heuristic, for adaptively defending 6G links against jamming and eavesdropping.

## Overview

The project models a 6G physical-layer-security (PLS) scenario: a Base Station transmitting to a legitimate user in the presence of jammers and eavesdroppers, with the wireless channel captured by a physically-grounded model (log-distance path loss, log-normal shadowing, Rician fading on the legitimate link, Rayleigh fading on adversary links). An RL agent learns to adapt transmit power, beamforming directivity, and artificial-noise injection in real time to maximize the secrecy capacity `Rs = [Cb − Ce]⁺` while controlling energy use and bit error rate.

The codebase has two generations:

- **`code/v1.ipynb`** — the original conference-paper prototype. A single DQN agent on a simplified, normalized-float toy environment (`Secure6GEnv`), compared against a random-action baseline and a hand-written threshold heuristic (TBHP).
- **`code/v2.ipynb`** — the journal-expansion codebase behind the current paper. A physically-grounded environment (`PLSEnv`) benchmarking four RL algorithms (DDPG, PPO, DQN, tabular Q-Learning) against the same two baselines, across multi-adversary scenarios and a reward-weight ablation study, merged into a single Colab-ready cell.

`v2` is the active codebase; `v1` is kept for reference as the original prototype behind the conference paper.

## Repository Structure

```
.
├── code/
│   ├── v1.ipynb                          # Conference prototype: single DQN vs. static/TBHP baselines on a toy env
│   └── v2.ipynb                          # Journal expansion: PLSEnv + DDPG/PPO/DQN/Q-Learning + ablation + multi-adversary experiments
├── paper/
│   ├── Continuous_DRL_for_6G_Security.pdf    # Current journal paper (v2 results)
│   ├── paper.pdf                             # Original submitted conference paper (v1 results)
│   └── Revision Report_*.pdf                 # Revision report for the conference paper
├── citation-papers/                      # Six reference papers on 6G PLS and DRL
├── presentation/                         # Slide decks
└── website/                              # Astro-based project website
```

## Environment

### v1 — `Secure6GEnv` (prototype)

- **State** (4 continuous values, normalized to `[0,1]`): signal strength, jammer power, noise level, Eve distance
- **Actions** (5 discrete): increase/decrease transmit power, beam left/right, inject artificial noise
- **Reward**: `secrecy_rate × 5 − jammer_power − noise_level × 0.5`, with dynamics defined by hand-picked arithmetic rather than physical channel models

### v2 — `PLSEnv` (journal expansion)

A config-driven (`PLSEnvConfig`) MDP grounded in PLS theory:

- **Channel model**: log-distance path loss with log-normal shadowing (Eq. 1); Rician fading for the legitimate BS→user link (LOS); Rayleigh fading for jammer/eavesdropper links (NLOS)
- **State** (fixed 7-D vector, independent of adversary count): `[ḡ_b, Ī_jam, P̄_jam, d̄_jam, Γ̄_e, d̄_eve, τ]` — legitimate channel gain, aggregate jammer interference, average jammer power, normalized distances to nearest jammer/eavesdropper, combined eavesdropper SINR, and normalized episode time
- **Actions**: `[Pt, w, Nt]` — transmit power, beamforming focus, artificial-noise power — continuous for DDPG/PPO, discretized into an `M³` grid for DQN/tabular Q-Learning
- **Reward**: `R = α·Rs − β·Energy − γ_ber·BER`, where `Rs = [Cb − Ce]⁺` is the Shannon secrecy capacity, `Energy = Pt/Pt_max + Nt/Nt_max`, and `BER = 0.5·erfc(√Γb)` is the BPSK bit-error rate on the legitimate link
- **Adversary scaling**: up to 3 simultaneous jammers and 3 eavesdroppers, with maximal-ratio-combining (MRC) modeling colluding eavesdroppers vs. a conservative worst-case single eavesdropper when non-colluding

## Getting Started

### Prerequisites

```bash
# v1
pip install stable-baselines3 gymnasium matplotlib numpy

# v2
pip install numpy scipy matplotlib torch
```

### Run v1 (prototype)

Open `code/v1.ipynb` and run cell by cell:

1. **Cells 1–2** — Install dependencies and import libraries
2. **Cell 3** — Define the `Secure6GEnv` Gymnasium environment
3. **Cells 4–5** — Train the DQN agent for 5,000 steps
4. **Cells 6–8** — Plot reward curves and run baseline comparisons
5. **Cells 9–10** — Print average rewards and compute improvement percentages
6. **Cells 11–12** — Track and plot secrecy rate over time
7. **Cells 13–14** — Visualize the static and dynamic network topology

### Run v2 (journal expansion)

`code/v2.ipynb` is a single cell — no multi-step notebook flow. In Colab: paste the cell's contents, optionally edit the `CONFIG` block at the top (`TRAIN_STEPS`, `SEEDS`, `EPISODE_LEN`, and the `RUN_MAIN_COMPARISON` / `RUN_ABLATION` / `RUN_MULTI_AGENT` toggles), then run it. It executes up to three experiments (matching the paper's Sections 4.2–4.4):

- **Main comparison** — DDPG vs. PPO vs. DQN vs. tabular Q-Learning vs. static/TBHP baselines, averaged over 5 seeds
- **Multi-adversary scalability** — escalating jammer/eavesdropper counts, up to 3 colluding eavesdroppers
- **Reward-weight ablation** — sweeps `beta_energy` and `gamma_ber` to test policy sensitivity/stability

Figures display inline and are saved as PNGs, and result tables are saved as `.csv`/`.md`, all under `results/`. If run in actual Colab, `results/` is zipped and downloaded automatically at the end. `argparse` was deliberately replaced with a plain `CONFIG` block, since Colab/Jupyter inject their own kernel-launch arguments into `sys.argv`. The paper's reported runs used `TRAIN_STEPS=1500`, `SEEDS=5`.

## Key Results

### v1 prototype (`Secure6GEnv`, single seed)

| Policy | Avg. Reward |
|--------|-------------|
| DQN (Ours) | **4.09** |
| TBHP | 2.89 |
| Static Random | 2.51 |

DQN outperforms the static baseline by **62.93%** and the heuristic baseline by **41.39%**.

### v2 journal results (`PLSEnv`, 5 seeds, base scenario: 1 jammer / 1 eavesdropper)

| Method | Avg Reward | Secrecy Rate (bps/Hz) | BER | vs Static | vs TBHP |
|--------|-----------:|-----------------------:|-----:|----------:|--------:|
| DDPG (continuous) | **13.29** | **3.02** | **0.035** | **+110.1%** | **+67.2%** |
| DQN (discrete) | 9.81 | 2.28 | 0.051 | +55.1% | +23.4% |
| Q-Learning (discrete) | 9.02 | 2.02 | — | +42.7% | +13.5% |
| TBHP (heuristic) | 7.95 | 1.84 | — | +25.7% | — |
| PPO (continuous) | 7.24 | 1.65 | 0.069 | +14.5% | −8.9% |
| Static Random | 6.32 | 1.48 | 0.084 | — | −20.4% |

DDPG — an off-policy continuous-control method — achieves the highest secrecy rate and lowest BER by avoiding the action-quantization loss that limits DQN/Q-Learning. PPO underperforms DQN in this setting because its on-policy updates struggle to adapt quickly to the high-variance Rayleigh/Rician fading transitions.

**Multi-adversary scalability**: as adversarial density increases from 1 jammer/1 eve to the worst case (3 jammers + 3 colluding eavesdroppers via MRC), DDPG's advantage widens further, ending at a **+143.1%** improvement over TBHP where DQN falls to only +24.6% and briefly drops *below* the TBHP baseline (−19.9%) in the 1-jammer/3-colluding-eves case — continuous control's fine-grained beamforming/noise adjustment is decisive against MRC-colluding threats that discrete quantization cannot precisely counter.

**Reward-weight ablation**: sweeping the energy penalty (`β = 0.5 → 2.0`) and BER penalty (`γ_ber = 0.25 → 1.0`) shows the agent converging to a stable, non-collapsing policy at every setting — increasing either penalty trades peak secrecy rate for lower energy use / BER rather than causing training instability, confirming the reward formulation is robust rather than a tuned artifact.

Full tables (Table 2–4) and figures are in `paper/Continuous_DRL_for_6G_Security.pdf`.

## References

See `citation-papers/` for the six papers this work builds on, covering GAN-powered DRL for IoT security, STAR-RIS-assisted multi-user ISAC, RL-based cross-layer security for 6G, covert communication with relay/RIS optimization, and advanced 6G security frameworks. A fuller literature comparison (Table 1) is in `paper/Continuous_DRL_for_6G_Security.pdf`.
