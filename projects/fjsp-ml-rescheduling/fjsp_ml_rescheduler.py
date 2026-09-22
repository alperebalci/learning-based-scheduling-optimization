"""Research-inspired Flexible Job-Shop (FJSP) rescheduling demo.

Combines a genetic algorithm scheduler with a Random Forest classifier for
rescheduling decisions under simulated processing-time variation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Operation:
    id: int
    job_id: int
    operation_number: int
    processing_times: Dict[int, float]
    compatible_machines: Set[int]
    required_config: int


@dataclass
class Job:
    id: int
    operations: List[Operation]
    priority: float = 1.0


@dataclass
class Machine:
    id: int
    available_configs: Set[int]
    setup_times: Dict[Tuple[int, int], float]


@dataclass
class Schedule:
    machine_schedules: Dict[int, List[Tuple[int, float, float]]]
    operation_assignments: Dict[int, Tuple[int, float, float]]
    makespan: float


class FJSPInstance:
    def __init__(self, jobs: List[Job], machines: List[Machine], num_workers: int = 1):
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if not jobs or not machines:
            raise ValueError("jobs and machines must be non-empty")

        self.jobs = jobs
        self.machines = {m.id: m for m in machines}
        self.num_workers = num_workers
        self.operations = [op for job in jobs for op in job.operations]
        self.operation_dict = {op.id: op for op in self.operations}

        if len(self.operation_dict) != len(self.operations):
            raise ValueError("operation ids must be unique")

        for machine in machines:
            if not machine.available_configs:
                raise ValueError(f"machine {machine.id} has no configuration")

        for op in self.operations:
            valid = op.compatible_machines & self.machines.keys()
            if not valid:
                raise ValueError(f"operation {op.id} has no compatible machine")
            if not any(op.required_config in self.machines[m].available_configs for m in valid):
                raise ValueError(f"operation {op.id} required config is unavailable")
            if set(op.processing_times) != set(op.compatible_machines):
                raise ValueError(f"operation {op.id} processing_times mismatch")

    def precedence(self) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = defaultdict(list)
        for job in self.jobs:
            for prev, cur in zip(job.operations, job.operations[1:]):
                result[cur.id].append(prev.id)
        return result


class GeneticAlgorithm:
    def __init__(self, instance: FJSPInstance, population_size: int = 120,
                 crossover_prob: float = 0.86, mutation_prob: float = 0.30):
        self.instance = instance
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.precedence = instance.precedence()
        self.jobs_by_id = {j.id: j for j in instance.jobs}

    def create_chromosome(self) -> List[int]:
        chromosome = [job.id for job in self.instance.jobs for _ in job.operations]
        random.shuffle(chromosome)
        return chromosome

    def fix_feasibility(self, chromosome: List[int]) -> List[int]:
        required = {j.id: len(j.operations) for j in self.instance.jobs}
        counts = defaultdict(int)
        fixed: List[int] = []
        for job_id in chromosome:
            if job_id in required and counts[job_id] < required[job_id]:
                fixed.append(job_id)
                counts[job_id] += 1
        for job_id, total in required.items():
            fixed.extend([job_id] * (total - counts[job_id]))
        return fixed

    def decode(self, chromosome: List[int]) -> Schedule:
        chromosome = self.fix_feasibility(chromosome)
        machine_schedules = {m: [] for m in self.instance.machines}
        machine_available = {m: 0.0 for m in self.instance.machines}
        machine_config = {
            m: min(machine.available_configs)
            for m, machine in self.instance.machines.items()
        }
        worker_available = [0.0] * self.instance.num_workers
        assignments: Dict[int, Tuple[int, float, float]] = {}
        job_counters = defaultdict(int)

        for job_id in chromosome:
            job = self.jobs_by_id[job_id]
            op = job.operations[job_counters[job_id]]
            job_counters[job_id] += 1

            precedence_ready = 0.0
            for pred_id in self.precedence[op.id]:
                precedence_ready = max(precedence_ready, assignments[pred_id][2])

            best = None
            for machine_id in sorted(op.compatible_machines & self.instance.machines.keys()):
                machine = self.instance.machines[machine_id]
                if op.required_config not in machine.available_configs:
                    continue

                setup = 0.0
                if machine_config[machine_id] != op.required_config:
                    setup = machine.setup_times.get(
                        (machine_config[machine_id], op.required_config), 1.0
                    )

                worker_id = None
                setup_end = None
                if setup > 0:
                    worker_id = min(
                        range(len(worker_available)), key=lambda i: worker_available[i]
                    )
                    setup_start = max(machine_available[machine_id], worker_available[worker_id])
                    setup_end = setup_start + setup
                    start = max(setup_end, precedence_ready)
                else:
                    start = max(machine_available[machine_id], precedence_ready)

                candidate = (start, machine_id, worker_id, setup_end)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate

            if best is None:
                raise ValueError(f"operation {op.id} cannot be assigned")

            start, machine_id, worker_id, setup_end = best
            end = start + op.processing_times[machine_id]
            if worker_id is not None and setup_end is not None:
                worker_available[worker_id] = setup_end

            machine_available[machine_id] = end
            machine_config[machine_id] = op.required_config
            machine_schedules[machine_id].append((op.id, start, end))
            assignments[op.id] = (machine_id, start, end)

        return Schedule(machine_schedules, assignments, max(machine_available.values()))

    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        if len(p1) < 3 or len(p1) != len(p2):
            return p1.copy(), p2.copy()
        a = random.randint(1, len(p1) - 2)
        b = random.randint(a + 1, len(p1) - 1)
        return (
            self.fix_feasibility(p1[:a] + p2[a:b] + p1[b:]),
            self.fix_feasibility(p2[:a] + p1[a:b] + p2[b:]),
        )

    def mutate(self, chromosome: List[int]) -> List[int]:
        if len(chromosome) < 2:
            return chromosome.copy()
        child = chromosome.copy()
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
        return self.fix_feasibility(child)

    @staticmethod
    def tournament(population: List[Tuple[List[int], float]], size: int = 3) -> List[int]:
        sample = random.sample(population, min(size, len(population)))
        return min(sample, key=lambda x: x[1])[0]

    def evolve(self, generations: int = 60) -> Schedule:
        population = []
        for _ in range(self.population_size):
            chromosome = self.create_chromosome()
            population.append((chromosome, self.decode(chromosome).makespan))

        best = min(population, key=lambda x: x[1])
        for _ in range(generations):
            new_population = [best]
            while len(new_population) < self.population_size:
                p1, p2 = self.tournament(population), self.tournament(population)
                if random.random() < self.crossover_prob:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                if random.random() < self.mutation_prob:
                    c1 = self.mutate(c1)
                if random.random() < self.mutation_prob:
                    c2 = self.mutate(c2)
                for child in (c1, c2):
                    if len(new_population) < self.population_size:
                        new_population.append((child, self.decode(child).makespan))
            population = new_population
            best = min(best, min(population, key=lambda x: x[1]), key=lambda x: x[1])
        return self.decode(best[0])


class ReschedulingClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def features(self, instance: FJSPInstance, schedule: Schedule, current_time: int,
                 variations: Dict[int, float], op_num: int = 5) -> np.ndarray:
        ranked = sorted(
            ((op.id, len(op.compatible_machines) / len(instance.machines))
             for op in instance.operations),
            key=lambda x: x[1],
            reverse=True,
        )[:op_num]
        values: List[float] = [float(current_time)]
        for op_id, flexibility in ranked:
            _, start, end = schedule.operation_assignments[op_id]
            remaining = max(0.0, end - current_time) if current_time >= start else end - start
            values.extend([remaining, variations.get(op_id, 0.0), flexibility])
        values.extend([0.0] * (1 + 3 * op_num - len(values)))
        return np.asarray(values, dtype=float)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, x: np.ndarray) -> bool:
        return bool(self.model.predict([x])[0])

    def probability(self, x: np.ndarray) -> float:
        return float(self.model.predict_proba([x])[0][1])


class Industry4Simulator:
    def __init__(self, instance: FJSPInstance):
        self.instance = instance
        self.ga = GeneticAlgorithm(instance)
        self.classifier = ReschedulingClassifier()

    def processing_variations(self, scenario_id: int) -> Dict[int, float]:
        rng = np.random.default_rng(scenario_id)
        result = {}
        for op in self.instance.operations:
            result[op.id] = float(rng.uniform(0.0, 0.20) if rng.random() < 0.7
                                  else rng.uniform(-0.15, 0.0))
        return result

    @staticmethod
    def is_beneficial(current: Schedule, candidate: Schedule, threshold: float = 0.05) -> bool:
        if current.makespan <= 0:
            return False
        return (current.makespan - candidate.makespan) / current.makespan >= threshold

    def training_data(self, scenarios: int = 20, horizon: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for scenario in range(scenarios):
            initial = self.ga.evolve(generations=25)
            variations = self.processing_variations(scenario)
            for t in range(0, horizon, 5):
                X.append(self.classifier.features(self.instance, initial, t, variations))
                candidate = self.ga.evolve(generations=15)
                y.append(int(self.is_beneficial(initial, candidate)))
        return np.asarray(X), np.asarray(y)

    def train_classifier(self, scenarios: int = 20) -> float:
        X, y = self.training_data(scenarios=scenarios)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("training labels contain only one class")
        stratify = y if counts.min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=stratify
        )
        self.classifier.train(X_train, y_train)
        if len(np.unique(y_test)) < 2:
            return float("nan")
        scores = [self.classifier.probability(x) for x in X_test]
        return float(roc_auc_score(y_test, scores))

    def run_ml(self, horizon: int = 50, scenario_id: int = 0) -> Tuple[List[float], int]:
        schedule = self.ga.evolve(generations=25)
        variations = self.processing_variations(scenario_id)
        history, count = [], 0
        for t in range(0, horizon, 2):
            x = self.classifier.features(self.instance, schedule, t, variations)
            if self.classifier.predict(x):
                candidate = self.ga.evolve(generations=15)
                if self.is_beneficial(schedule, candidate):
                    schedule, count = candidate, count + 1
            history.append(schedule.makespan)
        return history, count

    def run_periodic(self, period: int, horizon: int = 50) -> Tuple[List[float], int]:
        schedule = self.ga.evolve(generations=25)
        history, count = [], 0
        for t in range(0, horizon, 2):
            if t > 0 and t % period == 0:
                candidate = self.ga.evolve(generations=15)
                if self.is_beneficial(schedule, candidate):
                    schedule, count = candidate, count + 1
            history.append(schedule.makespan)
        return history, count


def create_sample_instance() -> FJSPInstance:
    machines = [
        Machine(
            id=i,
            available_configs={1, 2, 3, 4},
            setup_times={(a, b): (0.0 if a == b else 1.0 + 0.5 * abs(a - b))
                         for a in range(1, 5) for b in range(1, 5)},
        )
        for i in range(6)
    ]
    jobs = []
    for job_id in range(4):
        ops = []
        for op_num in range(3):
            compatible = set(random.sample(range(6), random.randint(2, 4)))
            ops.append(Operation(
                id=job_id * 3 + op_num,
                job_id=job_id,
                operation_number=op_num,
                processing_times={m: random.uniform(3, 15) for m in compatible},
                compatible_machines=compatible,
                required_config=random.randint(1, 4),
            ))
        jobs.append(Job(job_id, ops))
    return FJSPInstance(jobs, machines, num_workers=1)


def run_comparison() -> None:
    simulator = Industry4Simulator(create_sample_instance())
    auc = simulator.train_classifier(scenarios=20)
    print(f"Classifier AUC: {auc:.3f}" if not np.isnan(auc) else "Classifier AUC: unavailable")

    methods = {
        "ML": lambda: simulator.run_ml(),
        "P-2": lambda: simulator.run_periodic(2),
        "P-4": lambda: simulator.run_periodic(4),
        "P-7": lambda: simulator.run_periodic(7),
    }
    print("\nApproach  Reschedules  Improvement")
    for name, fn in methods.items():
        history, count = fn()
        improvement = 100 * (history[0] - history[-1]) / history[0] if history[0] else 0.0
        print(f"{name:<8} {count:<11} {improvement:>9.2f}%")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_comparison()
