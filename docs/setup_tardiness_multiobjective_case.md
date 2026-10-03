# Case Note: Sequence-Dependent Setups vs. Delivery Performance

## Problem statement

A common manufacturing scheduling conflict appears when switching between product families requires setup time.

Grouping identical or similar products reduces setup losses, but aggressive batching can delay other customer orders. The scheduler therefore trades off **production efficiency** against **delivery performance**.

A useful abstraction is:

- jobs/orders: (j \in J);
- product family of job (j): (g_j);
- processing time: (p_j);
- due date: (d_j);
- sequence-dependent setup time from family (a) to (b): (s_{ab});
- completion time: (C_j);
- tardiness: (T_j = \max(0, C_j-d_j));
- priority/penalty weight: (w_j).

Two core objectives are then:

[
f_1 = \sum \text{setup time or setup cost}
]

[
f_2 = \sum_j w_j T_j
]

Depending on the business, (f_2) may instead be replaced or supplemented by:

- number of tardy jobs;
- maximum tardiness;
- on-time-delivery (OTD) rate;
- service-level penalties;
- earliness/tardiness cost.

Avoid adding "customer satisfaction" as an independent objective unless it is measured separately. If it is merely a transformation of tardiness or OTD, treating both as objectives double-counts the same business effect.

Likewise, throughput should be included only when it captures a distinct operational objective rather than repeating the effect of makespan or utilization.

---

## This is a multi-objective scheduling problem

The important engineering question is not "which algorithm is AI?" but:

> Do we need one compromise schedule, or a set of efficient trade-off schedules?

### Weighted-sum formulation

If business trade-offs are already known, solve a scalarized objective such as:

[
\min \; \alpha \cdot \text{SetupCost} + \beta \cdot \text{WeightedTardiness}
]

This is operationally simple, but the result depends strongly on (alpha) and (eta).

Normalize objectives before weighting when their scales differ materially.

### Pareto formulation

If management wants to inspect alternatives, treat setup cost and delivery performance as separate objectives.

A schedule is Pareto-dominated if another feasible schedule is no worse on every objective and strictly better on at least one.

The output is a **Pareto frontier**, not a single "best" schedule.

This is where NSGA-II and related multi-objective metaheuristics are particularly useful.

---

## DEAP / NSGA-II

DEAP is a reasonable framework for implementing evolutionary multi-objective scheduling.

Suitable algorithms include:

- NSGA-II;
- NSGA-III for many-objective variants;
- SPEA2;
- custom evolutionary algorithms with domain-specific operators.

For a permutation-based production schedule, chromosome design and variation operators matter more than simply selecting a GA library.

Useful design choices:

- sequence-preserving crossover;
- swap, insertion, inversion, or block mutation;
- family-aware mutation to explore batching;
- due-date-aware mutation to recover service performance;
- repair operators for machine eligibility, precedence, calendars, or resource constraints.

### What DEAP can and cannot claim

DEAP/NSGA-II can search for high-quality non-dominated schedules.

It does **not** generally prove global optimality.

Use language such as:

- "Pareto-approximate solutions";
- "best non-dominated schedules found within the search budget";
- "heuristic trade-off frontier".

Do not call the result "optimal" unless an exact method or a valid bound establishes that claim.

---

## Exact and constraint-based methods

Before using GA or RL, establish a strong OR baseline.

### CP-SAT / Constraint Programming

A strong fit when the model contains:

- machine capacity;
- precedence;
- alternative machines;
- sequence-dependent setup times;
- release dates;
- due dates;
- calendars;
- forbidden transitions;
- logical constraints.

CP-SAT can optimize a scalarized objective and may provide optimality information or bounds, depending on the formulation and runtime.

For multi-objective analysis, common strategies are:

- repeated solves over different weights;
- epsilon-constraint method;
- lexicographic objectives;
- fixing one objective within a bound while optimizing another.

### MILP

MILP is attractive when:

- sequencing can be expressed with binary precedence variables;
- setup transitions are linearizable;
- business penalties are naturally linear;
- solver bounds and optimality gaps are important.

For large sequence-dependent scheduling problems, formulation strength and symmetry control are critical.

---

## RL formulation

RL is justified primarily when scheduling is **dynamic and repeatedly re-decided**.

Examples:

- new orders arrive online;
- processing times are stochastic;
- machines fail;
- rush orders appear;
- due dates or priorities change;
- setup state evolves during execution.

A typical MDP formulation is:

### State

- current machine/product family;
- queue of waiting jobs;
- remaining processing times;
- due-date slack;
- machine availability;
- setup matrix/context;
- current time;
- disturbance indicators.

### Action

Choose the next feasible:

- job;
- job-machine pair;
- dispatching rule;
- batch;
- rescheduling action.

### Reward

A scalar reward might penalize incremental:

[
r_t = -\lambda_s \Delta \text{SetupCost}_t
      -\lambda_T \Delta \text{Tardiness}_t
      -\lambda_i \Delta \text{IdleCost}_t
]

Hard constraints should be enforced by the environment or **action masking**, not learned only through penalties.

### Algorithms

- DQN variants for smaller discrete action spaces;
- PPO / actor-critic methods for larger masked discrete policies;
- GNN + PPO/DQN when the state is naturally a job-machine graph;
- offline RL when historical trajectories or solver-generated demonstrations are available.

SAC is primarily designed for continuous actions and is not the default choice for ordinary discrete sequencing decisions.

### Multi-objective RL

A single weighted reward collapses multiple objectives into one compromise.

If the goal is to recover a Pareto set rather than one fixed trade-off, use:

- multiple policies for different weight vectors;
- conditioned policies with preference vectors;
- multi-objective RL methods;
- RL for online adaptation around an offline Pareto plan.

RL does not provide a global scheduling-optimality guarantee.

---

## A stronger hybrid architecture

For this setup-vs-delivery problem, a defensible architecture is:

[
\text{Forecasting / state estimation}
\rightarrow
\text{OR or Pareto optimizer}
\rightarrow
\text{execution simulator}
\rightarrow
\text{rolling-horizon repair or policy}
]

One practical decomposition is:

1. **Offline planning**
   - CP-SAT/MILP for strong reference solutions;
   - NSGA-II for broad Pareto exploration on large instances.

2. **Prediction**
   - processing-time distributions;
   - failure risk;
   - order-arrival or demand uncertainty.

3. **Execution**
   - rolling-horizon reoptimization;
   - fast dispatching or learned policy for local reaction.

4. **Feasibility**
   - hard constraints enforced by OR model, simulator, or action masks.

5. **Robustness testing**
   - discrete-event simulation;
   - frozen disruption scenarios;
   - out-of-distribution load and mix changes.

This is more defensible than assigning one algorithm to every layer.

---

## Recommended experimental comparison

At minimum, compare:

| Method | Role |
|---|---|
| EDD / ATC / Minimum Slack | transparent dispatching baselines |
| Family-batching heuristic | setup-oriented baseline |
| CP-SAT or MILP | strong OR baseline |
| NSGA-II | Pareto metaheuristic |
| Rolling-horizon OR | dynamic optimization baseline |
| PPO/DQN policy | dynamic learned controller, if justified |

Evaluate every method on the same frozen instances.

Report:

- total setup time/cost;
- weighted tardiness;
- OTD rate;
- number of tardy orders;
- makespan if relevant;
- feasibility rate;
- solve/decision latency;
- robustness under disruptions;
- hypervolume / spacing / coverage for Pareto methods.

For multi-objective algorithms, do not compare only one arbitrarily selected schedule. Compare frontier quality and then apply a declared business decision rule to select an operating point.

---

## Decision rule

Use the simplest method that satisfies the operational requirement.

- If the problem is **static and deterministic**, start with CP-SAT/MILP/CP or a strong metaheuristic.
- If the main need is **trade-off exploration**, use Pareto optimization such as NSGA-II, epsilon-constraint CP-SAT/MILP, or both.
- If the problem is **dynamic**, start with rolling-horizon OR and dispatching; add RL only when repeated adaptive decisions and latency justify it.
- If uncertainty can be predicted, use **ML + optimization** before replacing the scheduler with an end-to-end policy.
- If feasibility is safety- or contract-critical, keep hard constraints outside the learned policy whenever possible.

The central design principle is:

> **Optimize known structure; learn unknown structure; simulate the interaction between them.**
