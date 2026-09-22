# Neural Large Neighborhood Search for Job-Shop Scheduling

A hybrid ML + search project for classical Job-Shop Scheduling.

```text
feasible dispatch sequence
    ↓
candidate destroy blocks
    ↓
PyTorch block-improvement scorer
    ↓
selected destroy neighborhood
    ↓
exact block permutation repair
    ↓
improved feasible schedule
```

The neural model does not construct the final schedule directly. It predicts which local large neighborhood is most promising; the selected block is repaired exactly by exhaustive unique-permutation search.

## Schedule representation

A sequence contains each job id once per operation. Decoding the kth occurrence of job `j` schedules its kth operation at:

```text
max(job_ready_time, machine_ready_time)
```

Every valid count-preserving sequence therefore decodes to a precedence- and machine-feasible semi-active schedule.

## Training labels

For every candidate contiguous destroy block:

1. compute block features from the current schedule;
2. enumerate every unique permutation of that block;
3. decode each repaired sequence;
4. label the block by relative best makespan improvement.

The block scorer is a small independently implemented PyTorch MLP trained on these exact local improvement labels.

## Baselines

- initial MWKR dispatch sequence;
- Random LNS: randomly select a block then exact-repair it;
- Learned LNS: score blocks and exact-repair the highest-ranked candidates;
- Oracle-block LNS: evaluate every block's exact repair before choosing the best one.

Oracle-block LNS is information-advantaged and is used as a local upper reference, not a deployable learned policy.

## Development run

Seed-42, 4 jobs × 4 machines, block length 5:

```text
training examples       360
positive labels          38.1%
final training MSE      0.002783

method               mean Cmax    mean improvement
Initial MWKR            39.000        0.00%
Learned LNS             34.188       11.20%
Random LNS              34.500       10.19%
Oracle-block LNS        32.312       15.68%
```

The learned scorer slightly beat random block selection in this finite fixture but did not approach the oracle-block policy. No general learned-search superiority claim is made.

## Exact local repair

The declared neighborhood repair is exact: every unique permutation of the selected block is enumerated. A regression test independently reconstructs the same neighborhood and checks that `exact_repair_block()` returns its minimum makespan.

This exactness applies only to the selected finite block, not to the full JSSP.

## GitHub Actions validation

A GitHub-hosted Ubuntu runner validated the complete implementation on Python 3.12.14 with PyTorch 2.13.0+cpu and NumPy 2.5.2. The remote regression suite passed all **5/5 tests**.

The CI smoke configuration used 3 jobs × 3 machines, block length 4, 8 training instances, 8 scorer epochs, 4 held-out test instances and 3 LNS iterations. Runner-observed result:

```text
training examples=48
final training MSE=0.009862
positive-label-rate=0.354

method               mean Cmax    mean improvement
Initial MWKR            30.000         0.00%
Learned LNS             26.500         8.11%
Random LNS              27.250         7.97%
Oracle-block LNS        26.500         9.60%
```

The learned scorer beat random block selection slightly on this small runner-specific smoke fixture. Oracle-block LNS remained information-advantaged and achieved the larger mean percentage improvement. These CI numbers validate the end-to-end learned-neighborhood pipeline; they are not a general learned-search superiority claim.

## Tests

The suite checks:

- schedule decoding feasibility;
- exact repair versus independent explicit neighborhood enumeration;
- finite block features;
- real PyTorch training/gradient path;
- incumbent monotonicity for learned, random and oracle LNS.

## Run

```bash
pip install -r requirements.txt
python run_neural_lns_jssp.py --self-test
python -m unittest discover -s tests -v
python run_neural_lns_jssp.py
```

## Scope

Not claimed:

- global JSSP optimality;
- learned LNS always beats random LNS;
- the small exact block enumeration scales to arbitrarily large destroy neighborhoods;
- the synthetic random JSSPs represent a production benchmark suite.
