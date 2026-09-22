# Learning-Augmented Online Machine Scheduling

A learning-augmented algorithms project for **non-clairvoyant identical parallel-machine makespan scheduling**.

The central design goal is not simply to use an ML prediction. It is to use a prediction while retaining a worst-case algorithmic guarantee when that prediction is bad.

```text
job/process features
        ↓
processing-time predictor
        ↓
predicted processing times
        ↓
predicted-LPT priority order
        ↓
dynamic non-preemptive list scheduling
        ↓
makespan
```

The benchmark explicitly studies:

```text
consistency when predictions are accurate
smooth empirical degradation as predictions worsen
robustness when predictions are arbitrarily poor
```

## Industrial Engineering interpretation

Think of a shift with several identical production resources and a set of jobs ready at time zero.

Examples include stylized:

- identical CNC machines;
- parallel inspection stations;
- equivalent packaging lines;
- homogeneous compute/processing resources.

A job's observable attributes are known before production, but its exact processing time is not known until the operation finishes.

The scheduling objective is:

```text
minimize Cmax
```

the completion time of the last job.

This is a classical `P||Cmax` scheduling problem viewed through a non-clairvoyant / learning-augmented information model.

---

# Information model

For job `j`:

```text
p_j       true processing time
p_hat_j   predicted processing time
```

All jobs are available at time zero.

Before execution:

```text
features and p_hat_j are available
p_j is not available
```

When a running job completes, its completion event is naturally observed by the scheduler.

The learning-augmented policy does not query `p_j` when it creates the priority order.

The simulator necessarily uses the hidden actual duration to advance the clock, but that value is not exposed to the scheduling rule before the completion event.

---

# Learning-augmented predicted LPT

The policy sorts jobs by decreasing predicted processing time:

```text
p_hat_(1) >= p_hat_(2) >= ... >= p_hat_(n)
```

It then runs a standard dynamic list schedule:

```text
whenever a machine becomes idle:
    start the next unscheduled job in predicted-LPT order
```

This distinction matters.

The method does **not** commit every job to a machine using predicted machine loads.

Only the priority order trusts the prediction. Machine assignment adapts to actual observed job completion times.

---

# Why this is learning-augmented rather than prediction-only

For `m` identical machines, every list schedule satisfies the classical Graham bound:

```text
C_list / OPT <= 2 - 1/m
```

independent of the job order.

Predicted LPT is still a list schedule. Therefore arbitrary prediction errors cannot remove that worst-case bound.

A short proof for this repository's all-jobs-at-time-zero setting:

Let `j` be the job that finishes last and let `S_j` be its start time.

Because the schedule never idles a machine while unscheduled jobs remain:

```text
m * S_j <= sum_(k != j) p_k
```

Hence:

```text
Cmax
= S_j + p_j
<= (sum_k p_k)/m + (1 - 1/m)*p_j
```

and both:

```text
(sum_k p_k)/m <= OPT
p_j <= OPT
```

so:

```text
Cmax <= (2 - 1/m) * OPT
```

This is the **robustness** property.

---

# Consistency under exact predictions

If:

```text
p_hat_j = p_j
```

for every job, predicted LPT becomes classical LPT.

For `m >= 2`, the standard LPT guarantee is:

```text
C_LPT / OPT <= 4/3 - 1/(3m)
```

This is the project's exact-prediction **consistency** bound.

The code includes the classical tight two-machine fixture:

```text
processing times = [3, 3, 2, 2, 2]

OPT = 6
LPT = 7

ratio = 7/6
      = 4/3 - 1/(3*2)
```

No stronger exact-prediction claim is made.

---

# Tight robustness fixture

Bad predictions can make the priority order poor.

For two machines:

```text
actual      = [1, 1, 2]
predictions = [3, 2, 1]
```

the long true job is placed last in predicted order.

The resulting dynamic list schedule has:

```text
Cmax = 3
OPT  = 2

ratio = 3/2
```

which exactly reaches:

```text
2 - 1/2 = 3/2
```

So the robustness bound is not merely decorative.

---

# Prediction-only comparator

The repository deliberately includes a more brittle alternative:

`Prediction-only static partition`

It:

1. sorts jobs by predicted processing time;
2. assigns every job to the machine with the smallest **predicted** committed load;
3. commits those assignments before execution;
4. never repairs a bad predicted allocation.

This is useful because it separates:

```text
using predictions inside a robust algorithm
```

from:

```text
blindly optimizing the prediction
```

A regression fixture uses:

```text
actual      = [1, 1, 2, 8, 8]
predictions = [4, 3, 5, 2, 1]
m           = 2
```

The static prediction-only plan obtains:

```text
Cmax / OPT = 1.8
```

which is already worse than the dynamic list-scheduling guarantee:

```text
2 - 1/m = 1.5
```

The learning-augmented dynamic policy remains inside the list-scheduling bound on the same fixture.

---

# Baselines

The benchmark compares:

### Learning-augmented predicted LPT

ML predictions determine job priority; actual machine assignment remains dynamic.

### Prediction-only static partition

All machine assignments are committed using predicted processing times.

### Prediction-free fixed-order list

Jobs run in original input order. No prediction is used.

This is still a robust classical list schedule with the same `2 - 1/m` worst-case guarantee.

### Clairvoyant true LPT

Uses actual processing times to construct LPT order.

This is information-advantaged and is **not** the exact optimum.

Because LPT itself is a heuristic, an imperfect predicted order can occasionally beat true LPT on an individual instance by chance.

### Exact offline optimum

A separate exact branch-and-bound solver receives all actual processing times.

It is used only for evaluation.

---

# Exact makespan oracle

`exact_makespan_branch_and_bound()` solves the declared finite `P||Cmax` partitioning problem using:

- descending processing-time search;
- an LPT incumbent;
- average-load lower bound;
- current maximum-load bound;
- largest remaining-job bound;
- symmetric machine-load pruning.

It reports `OPTIMAL` only after complete branch-and-bound proof.

This exact solver is intended for small/medium evaluation fixtures, not large-scale production scheduling.

## Independent exhaustive oracle

A second solver enumerates all `m^n` assignments for tiny instances.

Regression tests cross-check branch-and-bound against this independent exhaustive oracle.

During development this cross-check found and forced correction of an overly aggressive symmetry-pruning rule. The final implementation keeps only equal-load symmetry pruning.

---

# ML processing-time predictor

The synthetic job generator creates observable process features and hidden integer processing times.

The hidden function contains:

- quantity-like effects;
- nonlinear complexity;
- feature interactions;
- route/process effects;
- heteroskedastic process variation;
- occasional long-running jobs.

An `ExtraTreesRegressor` is trained on separate synthetic production jobs:

```text
features -> predicted processing time
```

The scheduling guarantees do not depend on ExtraTrees being a good model.

The predictor can be replaced without changing the robust scheduling layer.

---

# Prediction-error metrics

Two error measures are reported.

Worst multiplicative error:

```text
eta =
max_j max(
    p_j / p_hat_j,
    p_hat_j / p_j
)
```

with:

```text
eta >= 1
```

and mean absolute log error:

```text
mean_j |log(p_hat_j / p_j)|
```

The repository does not claim a new theorem that directly maps these empirical metrics to a competitive ratio.

They are used for controlled **smoothness experiments**.

---

# Controlled error sweep

To isolate prediction quality from ML model quality:

```text
p_hat_j = p_j * exp(epsilon_j)

epsilon_j ~ Normal(0, sigma^2)
```

The same actual scheduling instances are used across noise levels.

As `sigma` increases, the priority information is gradually corrupted.

The experiment measures:

```text
prediction error
Cmax / exact OPT
maximum observed Cmax / OPT
```

for both the learning-augmented and prediction-only approaches.

The theoretical list bound is checked programmatically at every sweep point.

---

# Development benchmark

Fixed seed-42 local run:

```text
machines                  3
jobs / instance          10
training instances      160
test instances           35
ExtraTrees              120
error-sweep instances    25
sweep jobs                9
```

ML prediction RMSE:

```text
6.837
```

The theoretical constants for three machines are:

```text
robust list bound              1.666667
exact-prediction LPT bound     1.222222
```

Held-out result:

```text
method                                  mean Cmax   mean/OPT   max/OPT

Learning-augmented predicted LPT          117.343     1.0656     1.1944
Prediction-only static partition          126.086     1.1431     1.3511
Prediction-free fixed-order list          122.286     1.1102     1.3091
Clairvoyant true LPT                      117.714     1.0683     1.1429
```

Every generated schedule passed the feasibility audit.

Interpretation:

- the learned predictions improved the dynamic list policy over the fixed-order baseline on this fixture;
- the dynamic learning-augmented method was substantially more stable than the static prediction-only comparator;
- predicted LPT slightly beat true LPT in mean on this finite sample, which can occur because LPT is not the exact optimizer;
- no universal superiority claim is made from one synthetic seed.

---

# Controlled degradation result

```text
sigma   mean|log err|   mean eta   LA ratio   static ratio   fixed list

0.00       0.0000         1.000      1.0253       1.0253       1.1129
0.10       0.0768         1.200      1.0329       1.0503       1.1129
0.25       0.1989         1.565      1.0547       1.1253       1.1129
0.50       0.4017         2.603      1.0611       1.1898       1.1129
0.80       0.5855         4.788      1.1051       1.3255       1.1129
1.20       0.9696        12.254      1.1108       1.4922       1.1129
```

Maximum observed learning-augmented ratio at every noise level remained below the three-machine theoretical robustness limit `1.6667`.

This table is empirical evidence of graceful degradation for the synthetic fixture. It is not presented as a new smoothness theorem.

A particularly useful pattern is visible:

```text
high-quality prediction:
    learning-augmented policy approaches clairvoyant LPT

poor prediction:
    learning-augmented policy approaches ordinary robust list behavior

prediction-only static commitment:
    degrades much more strongly
```

---

# Validated GitHub Actions run

GitHub Actions run `33155277405` completed successfully on Ubuntu 24.04 / CPython 3.12.14 with:

```text
NumPy          2.5.2
scikit-learn   1.9.0
```

All 10 regression/exact-oracle tests passed.

The CI smoke used `3` machines, `8` jobs per instance, `60` training instances, `12` held-out test instances, and a 60-tree ExtraTrees predictor.

```text
ML test RMSE                          6.437
robust list-scheduling bound          1.666667
exact-prediction LPT bound            1.222222
```

Held-out CI smoke:

```text
method                                  mean Cmax   mean/OPT   max/OPT
Learning-augmented predicted LPT           92.833     1.0622     1.1333
Prediction-only static partition           96.583     1.1021     1.3571
Prediction-free fixed-order list           94.667     1.0858     1.2468
Clairvoyant true LPT                       92.083     1.0523     1.1333
```

All four methods had feasibility rate `1.000`.

Controlled CI error sweep:

```text
sigma   mean|log err|   mean eta   LA ratio   static ratio   fixed list   max LA
0.00       0.0000         1.000      1.0361       1.0361       1.1459     1.1447
0.25       0.2242         1.580      1.0503       1.1632       1.1459     1.0845
0.80       0.6009         4.512      1.1267       1.2096       1.1459     1.2500
1.20       0.9637        11.402      1.0935       1.4520       1.1459     1.1494
```

The maximum observed learning-augmented ratio remained well below the theoretical `1.666667` list-scheduling robustness bound at every CI sweep point. The static prediction-only comparator degraded much more strongly as prediction noise increased. These are runner-specific smoke results, not universal performance claims.

---

# Schedule feasibility audit

Every result is checked for:

- every job scheduled exactly once;
- valid machine id;
- nonnegative start time;
- correct actual processing duration;
- no machine overlap;
- reported makespan equal to the latest completion.

---

# Regression suite

The executable suite contains 10 tests covering:

1. deterministic positive instance generation;
2. exact branch-and-bound vs full assignment enumeration;
3. tight `2 - 1/m` bad-prediction robustness example;
4. tight exact-prediction LPT consistency example;
5. exhaustive small processing-time/order search for the list-scheduling bound;
6. explicit failure of the brittle static prediction-only comparator;
7. feasibility audits for every scheduling method;
8. positive ML processing-time predictions;
9. hand-checked multiplicative prediction error;
10. controlled zero-error sweep and robustness-bound verification.

---

# Run

Install:

```bash
pip install -r requirements.txt
```

Self-test:

```bash
python run_learning_augmented_scheduling.py --self-test
```

Regression tests:

```bash
python -m unittest discover -s tests -v
```

Development experiment:

```bash
python run_learning_augmented_scheduling.py \
  --seed 42 \
  --machines 3 \
  --jobs 10 \
  --train-instances 160 \
  --test-instances 35 \
  --trees 120 \
  --sweep-instances 25 \
  --sweep-jobs 9 \
  --sigmas 0,0.1,0.25,0.5,0.8,1.2
```

---

# Scope and claims

Proved/standard guarantee used by the repository:

```text
every dynamic list schedule:
Cmax / OPT <= 2 - 1/m
```

Therefore predicted-LPT retains this bound for arbitrary predictions.

When predictions are exact, predicted-LPT becomes classical LPT, for which:

```text
Cmax / OPT <= 4/3 - 1/(3m)
```

for `m >= 2`.

Exact computational claims:

- the evaluation branch-and-bound solver is exact for the declared finite instances;
- it is independently cross-checked against full assignment enumeration on tiny fixtures;
- all reported heuristic schedules are feasibility-audited.

Not claimed:

- the ExtraTrees predictor is optimal;
- the empirical error sweep proves a new competitive-ratio formula;
- predicted LPT always beats fixed-order list scheduling;
- predicted LPT always beats true LPT;
- the static prediction-only comparator has the same robustness bound;
- the synthetic processing-time generator represents a real factory;
- the small exact solver is a large-scale industrial scheduling engine.

---

# Research context

The project is independently implemented and is motivated by the learning-augmented scheduling literature.

Relevant references include:

- Graham, R. L. — classical list-scheduling and LPT approximation bounds for identical parallel machines.
- Bampis, Kononov, Lucarelli, Pascual — *Non-Clairvoyant Makespan Minimization Scheduling with Predictions*, ISAAC 2023.
- Zhao, Li, Zomaya — *Uniform Machine Scheduling with Predictions*, ICAPS 2022.
- Mitzenmacher, Vassilvitskii — Algorithms with Predictions / Beyond Worst-Case Analysis.

No source code or API structure from these works is copied.
