# Multi-Agent Decision Automation in 2v2 Air Combat

A decentralized continuous-control MARL framework demonstrating emergent coordination under partial observability.

---

## Overview

This project implements a decentralized multi-agent reinforcement learning framework for a partially observable 2v2 air combat environment.

Agents are trained using Independent Proximal Policy Optimization (IPPO) with recurrent (LSTM-based) policies to handle partial observability and non-stationary learning dynamics.

The environment models continuous 3-DoF aircraft dynamics with load-factor-based control inputs.

---

## Environment

- Continuous 3-DoF aircraft model  
- Load-factor-based control (`nx`, `nz`, bank angle)  
- Binary fire command with engagement threshold logic  
- Partially observable setting  
- No inter-agent communication  

**Observation dimension:** `(46,)`  
All features are normalized.

---

## Learning Framework

- Independent PPO (IPPO)  
- Recurrent actor (LSTM-based)  
- Separate critic networks  
- Parameter sharing across teammates  
- Fully decentralized execution  

### Training Characteristics

- From-scratch 2v2 learning  
- Deterministic heuristic baseline for evaluation  
- No curriculum pre-training  

---

## Reward Design

Reward shaping includes:

- Distance-based shaping  
- Orientation alignment (cosine-based reward)  
- Engagement success reward  
- Survival reward  
- Episode outcome bonus  

Designed to balance tactical positioning and engagement effectiveness.

---

## Baseline Opponent

A deterministic rule-based heuristic agent is used for:

- Stable early-stage training  
- Structured evaluation  
- Non-random baseline comparison  

Heuristic logic includes:

- Nearest opponent targeting  
- Pursuit alignment  
- Engagement envelope threshold  
- Simple evasive maneuver  
- Basic teammate collision avoidance  

---

## Results

Performance against heuristic baseline:

- **30% win rate**
- **5% loss rate**
- **65% draw rate**

Emergent behaviors observed:

- Implicit role differentiation  
- Coordinated engagement without communication  
- Continuous maneuver adaptation  

---

## Scalability & Extensions

- Architecture scalable to N-agent settings  
- Adaptable to UAV / robotics autonomy domains  
- Extendable to self-play fine-tuning  
- Communication protocol learning (future work)  
- Physics-consistent weapon modeling (future extension)

---

## Repository Structure
- /scenarios --> environment and wrapper codes for both 1v1 and 2v2 configurations
- /runs --> train and evaluation results (v6 is final model for 2v2)
- Run train_2v2_rppo_LSTM.py, then eval_2v2_rppo_LSTM.py to train a 2v2 configuration model either from scratch or fine tune on a previous model.
