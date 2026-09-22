import random
import unittest

import numpy as np

from fjsp_ml_rescheduler import GeneticAlgorithm, create_sample_instance


class FJSPSmokeTests(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_small_ga_returns_complete_schedule(self):
        instance = create_sample_instance()
        solver = GeneticAlgorithm(instance, population_size=20)
        schedule = solver.evolve(generations=5)

        self.assertEqual(
            len(schedule.operation_assignments),
            len(instance.operations),
        )
        self.assertGreater(schedule.makespan, 0)

    def test_multiple_setup_workers_are_supported(self):
        instance = create_sample_instance()
        instance.num_workers = 2
        solver = GeneticAlgorithm(instance, population_size=10)
        schedule = solver.evolve(generations=2)

        self.assertEqual(
            len(schedule.operation_assignments),
            len(instance.operations),
        )


if __name__ == "__main__":
    unittest.main()
