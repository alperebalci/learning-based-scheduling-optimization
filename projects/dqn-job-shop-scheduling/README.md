# DQN Job Shop Scheduling

Deep Q-Network (DQN) implementation for a small Job Shop Scheduling Problem (JSSP) in Python using TensorFlow/Keras.

The environment models job precedence constraints and machine availability independently, allowing operations on different machines to overlap when feasible. The DQN agent learns which job's next operation should be scheduled at each decision point.

## Objectives

The reward function encourages schedules that:

- minimize makespan,
- reduce machine idle time,
- improve machine utilization,
- complete jobs efficiently.

## Features

- Custom Job Shop Scheduling environment
- Independent machine availability times
- Job precedence constraints via job-ready times
- DQN with experience replay
- Separate target network
- Epsilon-greedy exploration
- Valid-action masking in action selection and Bellman targets
- Round-robin and Shortest Processing Time (SPT) baselines
- Training plots for reward, makespan, utilization, and exploration rate

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python job_shop_dqn.py
```

The script trains the DQN agent, evaluates the learned policy, compares it with round-robin and SPT baselines, and displays training plots.

## Model Structure

The Q-network uses fully connected layers:

- 64 ReLU units
- 64 ReLU units
- 32 ReLU units
- linear output layer with one Q-value per job

The state contains normalized job progress, job-ready times, machine-availability times, and job-completion indicators.

## Scheduling Logic

For the next operation of a selected job:

```text
start_time = max(job_ready_time, machine_available_time)
end_time   = start_time + processing_time
```

This preserves both job precedence and machine-capacity constraints while allowing parallel work on different machines.

## Baselines

Two simple policies are included for comparison:

1. Round Robin
2. Shortest Processing Time (SPT)

Lower makespan is better. Higher utilization generally indicates better use of available machine capacity.

## Notes

This repository is intended as an educational reinforcement-learning implementation rather than an industrial-scale JSSP solver. Larger instances typically require stronger state representations, more advanced RL algorithms, graph-based models, or specialized optimization methods.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- TensorFlow / Keras
