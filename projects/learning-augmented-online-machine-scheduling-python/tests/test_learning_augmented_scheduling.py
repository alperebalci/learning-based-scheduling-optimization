import itertools
import unittest
import numpy as np

from la_scheduling import (
    ProcessingTimePredictor,
    audit_schedule,
    controlled_error_sweep,
    exact_makespan_branch_and_bound,
    exhaustive_assignment_oracle,
    fixed_order_list,
    flatten_instances,
    generate_instance_batch,
    generate_parallel_machine_instance,
    lpt_consistency_bound,
    list_scheduling_robustness_bound,
    multiplicative_error,
    predicted_lpt,
    static_predicted_partition,
    true_lpt,
)
from la_scheduling.schedule import list_schedule


class LearningAugmentedSchedulingTests(unittest.TestCase):
    def test_generator_is_reproducible_and_positive(self):
        a = generate_parallel_machine_instance(seed=10, n_jobs=8)
        b = generate_parallel_machine_instance(seed=10, n_jobs=8)
        np.testing.assert_array_equal(a.actual, b.actual)
        np.testing.assert_array_equal(a.features, b.features)
        self.assertTrue(np.all(a.actual > 0))

    def test_exact_branch_and_bound_matches_full_assignment_enumeration(self):
        fixtures = [
            np.array([3, 3, 2, 2, 2]),
            np.array([7, 4, 6, 2, 5, 3]),
            np.array([9, 1, 8, 2, 7]),
        ]
        for p in fixtures:
            exact = exact_makespan_branch_and_bound(p, 2)
            brute = exhaustive_assignment_oracle(p, 2)
            self.assertEqual(exact.status, "OPTIMAL")
            self.assertEqual(exact.makespan, brute.makespan)

    def test_predicted_lpt_hits_tight_list_scheduling_bound_with_bad_prediction(self):
        actual = np.array([1, 1, 2])
        predicted = np.array([3.0, 2.0, 1.0])
        result = predicted_lpt(actual, predicted, 2)
        optimum = exact_makespan_branch_and_bound(actual, 2).makespan
        self.assertEqual(result.makespan, 3)
        self.assertEqual(optimum, 2)
        self.assertAlmostEqual(
            result.makespan/optimum,
            list_scheduling_robustness_bound(2),
            places=12,
        )

    def test_exact_predictions_recover_tight_lpt_consistency_example(self):
        actual = np.array([3, 3, 2, 2, 2])
        result = predicted_lpt(actual, actual.astype(float), 2)
        optimum = exact_makespan_branch_and_bound(actual, 2).makespan
        self.assertEqual(result.makespan, 7)
        self.assertEqual(optimum, 6)
        self.assertAlmostEqual(
            result.makespan/optimum,
            lpt_consistency_bound(2),
            places=12,
        )

    def test_exhaustive_small_orders_never_break_list_scheduling_bound(self):
        m = 2
        bound = list_scheduling_robustness_bound(m)
        for values in itertools.product([1, 2, 3], repeat=4):
            actual = np.asarray(values, dtype=np.int64)
            optimum = exact_makespan_branch_and_bound(actual, m).makespan
            for order in itertools.permutations(range(len(actual))):
                result = list_schedule(actual, order, m)
                self.assertLessEqual(result.makespan/optimum, bound + 1e-12)

    def test_prediction_only_static_partition_can_be_less_robust_than_dynamic_list(self):
        actual = np.array([1, 1, 2, 8, 8])
        predicted = np.array([4., 3., 5., 2., 1.])
        optimum = exact_makespan_branch_and_bound(actual, 2).makespan
        static = static_predicted_partition(actual, predicted, 2)
        dynamic = predicted_lpt(actual, predicted, 2)
        self.assertAlmostEqual(static.makespan/optimum, 1.8, places=12)
        self.assertLessEqual(
            dynamic.makespan/optimum,
            list_scheduling_robustness_bound(2) + 1e-12,
        )
        self.assertGreater(
            static.makespan/optimum,
            list_scheduling_robustness_bound(2),
        )

    def test_all_schedulers_return_feasible_nonoverlapping_schedules(self):
        instance = generate_parallel_machine_instance(seed=22, n_jobs=10)
        pred = instance.actual.astype(float) * np.linspace(0.7, 1.3, 10)
        methods = [
            predicted_lpt(instance.actual, pred, 3),
            fixed_order_list(instance.actual, 3),
            true_lpt(instance.actual, 3),
            static_predicted_partition(instance.actual, pred, 3),
        ]
        for result in methods:
            self.assertEqual(audit_schedule(instance.actual, result, 3), 0.0)

    def test_predictor_produces_positive_processing_time_predictions(self):
        train = generate_instance_batch(20, seed=30, n_jobs=8)
        test = generate_instance_batch(3, seed=31_000, n_jobs=8)
        x, y = flatten_instances(train)
        predictor = ProcessingTimePredictor.fit(x, y, seed=2, trees=40)
        for instance in test:
            pred = predictor.predict(instance.features)
            self.assertEqual(pred.shape, instance.actual.shape)
            self.assertTrue(np.all(pred > 0))

    def test_multiplicative_error_hand_check(self):
        actual = np.array([2., 4., 8.])
        pred = np.array([1., 8., 4.])
        self.assertAlmostEqual(multiplicative_error(actual, pred), 2.0)

    def test_controlled_zero_error_equals_true_lpt(self):
        instances = generate_instance_batch(5, seed=40, n_jobs=7)
        rows = controlled_error_sweep(
            instances,
            n_machines=3,
            sigmas=[0.0, 0.5],
            seed=9,
        )
        self.assertAlmostEqual(
            rows[0].predicted_lpt_ratio,
            rows[0].true_lpt_ratio,
            places=12,
        )
        self.assertEqual(rows[0].mean_log_error, 0.0)
        bound = list_scheduling_robustness_bound(3)
        self.assertTrue(all(r.max_predicted_lpt_ratio <= bound + 1e-12 for r in rows))


if __name__ == "__main__":
    unittest.main()
