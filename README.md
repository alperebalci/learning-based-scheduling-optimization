# Learning-Based Scheduling Optimization

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Included projects

- [`dqn-job-shop-scheduling`](projects/dqn-job-shop-scheduling/)
- [`dynamic-automotive-paint-shop-scheduling-rl`](projects/dynamic-automotive-paint-shop-scheduling-rl/)
- [`fjsp-ml-rescheduling`](projects/fjsp-ml-rescheduling/)
- [`job-shop-lib-rl-scheduling`](projects/job-shop-lib-rl-scheduling/)
- [`job-shop-scheduling-ppo`](projects/job-shop-scheduling-ppo/)
- [`learning-augmented-online-machine-scheduling-python`](projects/learning-augmented-online-machine-scheduling-python/)
- [`neural-large-neighborhood-search-job-shop-scheduling-pytorch`](projects/neural-large-neighborhood-search-job-shop-scheduling-pytorch/)
- [`reinforcement-learning-job-shop-scheduling-pytorch`](projects/reinforcement-learning-job-shop-scheduling-pytorch/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot.
<!-- portfolio-umbrella:end -->

Research-oriented Industrial Engineering / Operations Research benchmark studying whether policies learned **offline from OR-generated scheduling decisions** can approach strong optimization and dispatching baselines without online exploration.

## Research question

How sensitive are offline scheduling policies to demonstration quality, and can masked neural CQL/IQL provide a defensible quality/latency trade-off against behavior cloning, strong dispatching rules and CP-SAT on frozen unseen FJSP instances?

## Practitioner method-selection guide

For the broader industrial-engineering question — **when should scheduling use OR, ML, deep learning, reinforcement learning, or hybrid AI?** — see [`docs/method_selection_guide.md`](docs/method_selection_guide.md).

The guide uses an **OR-first, learning-where-justified** decision framework and covers CP-SAT/MILP, supervised prediction, GNNs/attention, DQN/PPO, offline RL (BC/CQL/IQL), neural LNS, and decision-focused learning.

A concrete manufacturing case on **sequence-dependent setup time vs. delivery performance** is documented in [`docs/setup_tardiness_multiobjective_case.md`](docs/setup_tardiness_multiobjective_case.md), including Pareto optimization with NSGA-II, exact/constraint-based baselines, and dynamic RL/hybrid formulations.

## Status

**Feature-complete research benchmark.**

The repository now includes the full experimental chain:

1. deterministic FJSP simulation and seeded instance generation;
2. transparent SPT, EDD and Minimum Slack dispatching baselines;
3. an OR-Tools CP-SAT expert;
4. candidate-level expert demonstrations;
5. mixed-quality offline trajectories with explicit behavior-policy provenance;
6. linear behavior cloning, CQL and IQL baselines;
7. masked PyTorch neural CQL/IQL;
8. dataset-quality ablation;
9. frozen nominal-final and structural-OOD blocks;
10. multi-seed neural training;
11. paired bootstrap confidence intervals and exact sign tests;
12. episode latency and feasibility accounting;
13. tests and GitHub Actions CI;
14. a documented final research protocol in `docs/final_report.md`.

No method is assumed to win. The benchmark is designed so CP-SAT, Minimum Slack or behavior cloning can remain the operational recommendation if the added complexity of offline RL is not justified.

## Decision problem

Each job contains precedence-constrained operations. Every operation is eligible on a subset of machines with machine-dependent processing times. At each decision epoch the controller chooses `(job_id, machine_id)` for the next unscheduled operation of a job.

Primary KPI: **weighted tardiness**.

Secondary KPIs:

- makespan;
- episode-level controller latency;
- feasibility rate.

## Offline data contract

Every logged transition stores:

- selected action features;
- all feasible **current-state** action features;
- all feasible next-state action features;
- terminal utility;
- done flag;
- behavior-policy name;
- instance seed.

The current feasible set is required for masked discrete CQL and prevents the conservative penalty from being evaluated on the wrong state support.

Trajectory utility is

`-(weighted_tardiness + 0.02 * makespan)`

with zero intermediate reward.

## Dataset-quality ablation

Four named data regimes are retained:

- `expert_only`: CP-SAT only;
- `expert_strong`: CP-SAT + Minimum Slack;
- `balanced`: CP-SAT + Minimum Slack + EDD + SPT;
- `corrupted`: balanced data + an explicitly random behavior policy.

Random data is never hidden inside an unlabeled mixture. Dataset provenance stays machine-readable.

## Learning stack

### Behavior cloning

A ridge-regularized candidate ranker trained only from CP-SAT action labels.

### Linear offline RL

Auditable fitted CQL and expectile-IQL baselines expose the value-learning objectives without neural-network complexity.

### Neural offline RL

`NeuralCQL` and `NeuralIQL` use compact PyTorch MLP scorers over variable-size feasible action sets. Policies score only actions that are feasible in the current FJSP state.

Neural support is optional:

```bash
pip install -e '.[dev,neural]'
```

## Frozen final evaluation

The final protocol is encoded in `configs/final_evaluation.json`.

Training instances:

- seeds `10-19`.

Independent neural training seeds:

- `0`, `1`, `2`.

Frozen evaluation blocks:

- nominal final: seeds `1000-1019`;
- OOD scale: seeds `1100-1109`, using 8 jobs, 5 machines and 4 operations/job;
- OOD flexibility: seeds `1200-1209`, using reduced machine eligibility.

These blocks must not be used for hyperparameter selection.

For learned controllers, model-seed results are averaged by instance before paired inference so training replicates are not treated as independent test instances.

## Statistical evaluation

Weighted-tardiness comparisons are paired by instance seed against Minimum Slack and report:

- mean paired difference;
- paired 95% bootstrap confidence interval;
- median difference;
- win rate;
- exact two-sided sign-test p-value;
- sample size.

Statistical significance alone is not treated as operational superiority. Feasibility, effect magnitude and latency are evaluated jointly.

## Latency and reliability

Episode latency is measured with `time.perf_counter()` around the complete controller execution.

For CP-SAT this includes optimization plus replay through the common simulator. For learned and heuristic controllers it includes repeated action selection and simulation.

A controller is not eligible for promotion unless feasibility is 100% on every frozen block.

## Reproduce

Base stack:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m offline_fjsp.experiment
python -m offline_fjsp.phase3_experiment
```

Neural ablation:

```bash
pip install -e '.[dev,neural]'
python -m offline_fjsp.phase4_experiment
```

Frozen final campaign:

```bash
python -m offline_fjsp.final_evaluation
```

The final command prints aggregate quality/latency/reliability results followed by paired statistical comparisons against Minimum Slack.

## Repository map

```text
src/offline_fjsp/
  model.py
  generator.py
  policies.py
  cpsat_expert.py
  features.py
  dataset.py
  offline_dataset.py
  behavior_cloning.py
  cql.py
  iql.py
  neural_offline.py
  statistics.py
  experiment.py
  phase3_experiment.py
  phase4_experiment.py
  final_evaluation.py
tests/
  test_core.py
  test_research_pipeline.py
  test_phase3.py
  test_phase4.py
  test_final_evaluation.py
configs/
  experiment.json
  final_evaluation.json
docs/
  experimental_protocol.md
  final_report.md
.github/workflows/
  ci.yml
```

## Scientific acceptance rules

- strong non-learning baselines precede learning claims;
- behavior-policy provenance is explicit;
- current and next feasible action support are logged separately;
- train, validation and frozen final seeds are separated;
- corrupted data is deliberate and labeled;
- learned methods are evaluated under multiple independent model seeds;
- final comparisons are paired by instance;
- confidence intervals, effect direction, latency and feasibility are all reported;
- negative/null results are retained;
- no controller is called superior solely because it is more complex.

## Interpretation

If CP-SAT remains clearly better and solve latency is operationally acceptable, CP-SAT is the recommendation.

If Minimum Slack remains competitive at much lower latency, the heuristic is the recommendation.

If behavior cloning matches CQL/IQL, the simpler imitation model is preferred.

Offline RL earns promotion only when its improvement survives frozen nominal and OOD evaluation and justifies its additional complexity.

## Scope boundary

The repository is complete for this research question. Graph neural policies, external industrial benchmark datasets, richer disruptions and digital-twin integration should be treated as separate follow-up studies rather than silently expanding this benchmark.

A concrete specification for one such study—**adaptive job-shop rescheduling with explicit feasibility, a learned constraint shield, and RL**—is available in [`docs/follow_up_adaptive_job_shop_constraint_rl.md`](docs/follow_up_adaptive_job_shop_constraint_rl.md).

## License

MIT
