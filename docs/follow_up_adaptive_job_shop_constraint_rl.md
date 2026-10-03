# Follow-up Study: Adaptive Job-Shop Rescheduling with Constraint Shielding and Reinforcement Learning

**Status:** Proposed research extension. This document is a project specification, not an implemented benchmark.

## Research question

Can a reinforcement-learning rescheduling policy react to machine failures, rush orders and processing-time changes while remaining inside a validated feasible operating region defined by explicit scheduling constraints and a learned operational-feasibility model?

## Motivation

The repository already contains strong scheduling baselines and learned scheduling policies. The missing research question is the hybrid interaction between:

- explicit scheduling feasibility;
- empirically learned operational constraints or expert acceptability;
- disruption-aware sequential decision making.

The purpose of this extension is not to replace CP-SAT/MILP. It is to test whether a learned policy can make low-latency rescheduling decisions when repeated exact reoptimization is expensive or operationally unstable.

## Hybrid architecture

### Layer 1 — explicit feasibility

Keep deterministic constraints explicit:

- operation precedence;
- machine eligibility;
- machine capacity;
- release times;
- maintenance/outage windows;
- no-overlap;
- frozen operations already in execution.

These constraints define the legal action set.

### Layer 2 — learned operational constraint

Train a calibrated model from historical or synthetic scheduling outcomes to estimate whether a locally feasible rescheduling action is operationally acceptable.

Possible learned signals:

- excessive downstream WIP;
- unstable setup patterns;
- congestion around shared tooling;
- expert rejection of otherwise legal schedules;
- service-risk patterns not captured by the base model.

The learned model acts as a shield/filter, not as a replacement for hard constraints.

### Layer 3 — RL rescheduling policy

Use DQN for a compact discrete dispatch/action set or PPO for a larger parameterized policy.

State may include:

- ready operations and due dates;
- machine status and remaining processing time;
- sequence-dependent setup state;
- WIP by work center;
- known failures and repair-time estimates;
- rush-order arrivals;
- learned-feasibility scores.

Action examples:

- select the next job-operation/machine assignment;
- choose among a restricted set of feasible local schedule repairs;
- select a dispatching rule for the next decision interval.

## Reward

A benchmark reward should reflect operational trade-offs rather than only makespan:

```text
reward =
- weighted tardiness
- setup/changeover cost
- WIP penalty
- rescheduling-instability penalty
- disruption recovery penalty
```

Hard infeasibility should be prevented by masking rather than merely assigned a large negative reward.

A learned-constraint violation can either be masked above a validated risk threshold or penalized while its intervention rate is reported explicitly.

## Disruption model

Evaluate on seeded scenarios containing:

- machine breakdowns;
- stochastic repair duration;
- rush-order insertion;
- processing-time perturbation;
- temporary machine unavailability;
- optional operator/tool shortages.

Training, validation and frozen final disruption seeds must be separated.

## Baselines

Compare at least:

- static dispatching rules: SPT, EDD, Minimum Slack;
- rolling-horizon CP-SAT or MILP;
- periodic full reoptimization;
- RL with hard action masking only;
- learned constraint shield + non-RL heuristic;
- RL + learned shield;
- optional oracle on very small fixtures.

The hybrid method should earn its complexity against both strong OR and simple dispatching baselines.

## Evaluation

Primary metrics:

- weighted tardiness;
- makespan;
- feasibility rate;
- decision latency.

Secondary metrics:

- setup/changeover cost;
- WIP;
- number of rescheduled operations;
- schedule instability relative to the pre-disruption plan;
- disruption recovery time;
- learned-shield intervention rate;
- false-feasible rate of the learned constraint model;
- objective gap to the rolling-horizon OR baseline.

Report paired results under common disruption seeds.

## Safety and validity rules

1. Hard scheduling feasibility is never learned.
2. A schedule is not called safe merely because the learned model assigns high probability.
3. Learned-shield quality and RL policy quality are reported separately.
4. If the shield intervenes frequently, constraint satisfaction cannot be attributed to the raw policy.
5. Final disruption scenarios are frozen before model selection.
6. Null results are retained if rolling-horizon OR remains better.
7. Real-time claims require measured end-to-end decision latency.

## Suggested experimental phases

### Phase 1 — small exact rescheduling fixture
Use a small FJSP/JSSP instance where a dynamic-programming or exact rolling-horizon reference is available.

### Phase 2 — disruption-aware simulator
Add failures, rush orders and processing-time perturbations with fixed seeds.

### Phase 3 — learned constraint shield
Train and calibrate the operational-feasibility model independently of the final evaluation block.

### Phase 4 — RL policy
Train DQN/PPO only over actions that pass explicit feasibility; compare shielded and unshielded variants.

### Phase 5 — industrial-scale benchmark
Move to larger instances where exact reoptimization is time-limited and report incumbent quality, gap, latency and fallback behavior.

## Relationship to this repository

This study is intentionally a follow-up rather than an expansion of the current feature-complete offline scheduling benchmark. It directly connects the repository's scheduling expertise with constraint learning and safe/hybrid RL while preserving the existing benchmark's frozen research question.
