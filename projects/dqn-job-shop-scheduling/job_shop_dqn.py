import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


class JobShopEnvironment:
    """
    Job Shop Scheduling environment.

    Each action selects a job and schedules that job's next operation.
    Operations belonging to different machines may overlap in time.

    Objective:
      - minimize makespan,
      - reduce machine idle time,
      - improve machine utilization.
    """

    def __init__(
        self,
        num_jobs=5,
        num_machines=3,
        processing_times=None,
        job_sequences=None,
        seed=42,
    ):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.rng = np.random.default_rng(seed)

        if processing_times is None:
            self.processing_times = self.rng.integers(
                1, 11, size=(num_jobs, num_machines)
            )
        else:
            self.processing_times = np.asarray(processing_times, dtype=int)

        if self.processing_times.shape != (num_jobs, num_machines):
            raise ValueError(
                "processing_times must have shape "
                f"({num_jobs}, {num_machines})"
            )

        if job_sequences is None:
            self.job_sequences = [
                self.rng.permutation(num_machines).tolist()
                for _ in range(num_jobs)
            ]
        else:
            self.job_sequences = [list(seq) for seq in job_sequences]

        if len(self.job_sequences) != num_jobs:
            raise ValueError("job_sequences must contain one sequence per job.")

        for seq in self.job_sequences:
            if len(seq) != num_machines:
                raise ValueError(
                    "Each job sequence must contain exactly num_machines entries."
                )
            if sorted(seq) != list(range(num_machines)):
                raise ValueError(
                    "Each job sequence must be a permutation of machine IDs."
                )

        self.total_required_processing = float(np.sum(self.processing_times))
        self.time_scale = max(self.total_required_processing, 1.0)

        self.reset()

    def reset(self):
        self.job_step = np.zeros(self.num_jobs, dtype=int)
        self.job_ready_time = np.zeros(self.num_jobs, dtype=float)
        self.machine_available = np.zeros(self.num_machines, dtype=float)
        self.job_completed = np.zeros(self.num_jobs, dtype=bool)

        self.machine_idle_time = np.zeros(self.num_machines, dtype=float)
        self.scheduled_processing_time = 0.0
        self.operation_log = []

        return self._get_state()

    def _get_state(self):
        progress_denominator = max(self.num_machines, 1)

        return np.concatenate(
            [
                self.job_step / progress_denominator,
                self.job_ready_time / self.time_scale,
                self.machine_available / self.time_scale,
                self.job_completed.astype(float),
            ]
        ).astype(np.float32)

    def get_valid_actions(self):
        return [
            job_id
            for job_id in range(self.num_jobs)
            if not self.job_completed[job_id]
        ]

    def step(self, action):
        if action < 0 or action >= self.num_jobs:
            raise ValueError(f"Action must be between 0 and {self.num_jobs - 1}.")

        if self.job_completed[action]:
            done = bool(np.all(self.job_completed))
            return self._get_state(), -10.0, done, self._get_info()

        job_id = action
        operation_idx = self.job_step[job_id]

        machine_id = self.job_sequences[job_id][operation_idx]
        processing_time = float(self.processing_times[job_id, machine_id])

        previous_makespan = self.makespan

        machine_ready = self.machine_available[machine_id]
        job_ready = self.job_ready_time[job_id]
        start_time = max(machine_ready, job_ready)
        end_time = start_time + processing_time

        idle_increment = max(0.0, start_time - machine_ready)
        self.machine_idle_time[machine_id] += idle_increment

        self.machine_available[machine_id] = end_time
        self.job_ready_time[job_id] = end_time
        self.scheduled_processing_time += processing_time

        self.operation_log.append(
            {
                "job": job_id,
                "operation": operation_idx,
                "machine": machine_id,
                "start": start_time,
                "end": end_time,
                "processing_time": processing_time,
            }
        )

        self.job_step[job_id] += 1
        if self.job_step[job_id] >= len(self.job_sequences[job_id]):
            self.job_completed[job_id] = True

        makespan_increment = self.makespan - previous_makespan
        reward = self._calculate_reward(
            makespan_increment=makespan_increment,
            idle_increment=idle_increment,
            completed_job=self.job_completed[job_id],
        )

        done = bool(np.all(self.job_completed))
        if done:
            reward += 20.0

        return self._get_state(), reward, done, self._get_info()

    @property
    def makespan(self):
        return float(np.max(self.machine_available))

    def _calculate_reward(
        self,
        makespan_increment,
        idle_increment,
        completed_job,
    ):
        reward = 1.0
        reward -= 0.5 * makespan_increment
        reward -= 0.2 * idle_increment

        if completed_job:
            reward += 2.0

        return float(reward)

    def _calculate_utilization(self):
        if self.makespan <= 0:
            return 0.0

        capacity = self.makespan * self.num_machines
        return float(self.scheduled_processing_time / capacity)

    def _get_info(self):
        return {
            "makespan": self.makespan,
            "total_idle_time": float(np.sum(self.machine_idle_time)),
            "utilization": self._calculate_utilization(),
        }


class DQNAgent:
    """Deep Q-Network agent with target-network and valid-action masking."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=5000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()

    def _build_model(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.state_size,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(self.action_size, activation="linear"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_valid_actions,
    ):
        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
                tuple(next_valid_actions),
            )
        )

    def act(self, state, valid_actions):
        if not valid_actions:
            raise ValueError("No valid actions are available.")

        if np.random.random() <= self.epsilon:
            return random.choice(valid_actions)

        q_values = self.q_network.predict(
            state.reshape(1, -1), verbose=0
        )[0]

        return max(valid_actions, key=lambda a: q_values[a])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=int)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=bool)
        next_valid_actions = [e[5] for e in batch]

        targets = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(
            next_states, verbose=0
        )

        for i in range(batch_size):
            if dones[i] or not next_valid_actions[i]:
                target_value = rewards[i]
            else:
                valid_next_q = [
                    next_q_values[i][a]
                    for a in next_valid_actions[i]
                ]
                target_value = rewards[i] + self.gamma * max(valid_next_q)

            targets[i][actions[i]] = target_value

        self.q_network.fit(
            states,
            targets,
            epochs=1,
            verbose=0,
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )


def train_dqn_scheduler(
    episodes=500,
    batch_size=32,
    target_update_frequency=25,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)

    env = JobShopEnvironment(seed=seed)
    state_size = len(env._get_state())
    action_size = env.num_jobs

    agent = DQNAgent(state_size, action_size)

    scores = []
    makespans = []
    utilizations = []

    print("Training DQN for Job Shop Scheduling")
    print("=" * 50)
    print(f"Jobs: {env.num_jobs}, Machines: {env.num_machines}")
    print(f"Processing Times:\n{env.processing_times}")
    print(f"Job Sequences: {env.job_sequences}")
    print("=" * 50)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0

        while True:
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)

            next_state, reward, done, info = env.step(action)
            next_valid_actions = (
                [] if done else env.get_valid_actions()
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
                next_valid_actions,
            )

            state = next_state
            total_reward += reward

            agent.replay(batch_size=batch_size)

            if done:
                scores.append(total_reward)
                makespans.append(info["makespan"])
                utilizations.append(info["utilization"])
                break

        if (episode + 1) % target_update_frequency == 0:
            agent.update_target_network()

        if episode % 100 == 0 or episode == episodes - 1:
            recent = slice(max(0, len(scores) - 100), len(scores))

            avg_score = np.mean(scores[recent])
            avg_makespan = np.mean(makespans[recent])
            avg_utilization = np.mean(utilizations[recent])

            print(f"Episode {episode}")
            print(f"  Avg Reward: {avg_score:.2f}")
            print(f"  Avg Makespan: {avg_makespan:.2f}")
            print(f"  Avg Utilization: {avg_utilization:.2%}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print("-" * 30)

    return agent, env, scores, makespans, utilizations


def test_trained_agent(agent, env, num_tests=10):
    print("\nTesting Trained Agent")
    print("=" * 30)

    test_makespans = []
    test_utilizations = []

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    try:
        for test in range(num_tests):
            state = env.reset()

            while True:
                valid_actions = env.get_valid_actions()
                action = agent.act(state, valid_actions)

                state, _, done, info = env.step(action)

                if done:
                    test_makespans.append(info["makespan"])
                    test_utilizations.append(info["utilization"])

                    if test == 0:
                        print("First test schedule:")
                        for op in sorted(
                            env.operation_log,
                            key=lambda x: (x["start"], x["machine"]),
                        ):
                            print(
                                f"  Job {op['job']} / Op {op['operation']} "
                                f"-> Machine {op['machine']} "
                                f"[{op['start']:.1f}, {op['end']:.1f}]"
                            )

                        print(f"Makespan: {info['makespan']:.2f}")
                        print(
                            "Total Idle Time: "
                            f"{info['total_idle_time']:.2f}"
                        )
                        print(
                            "Machine Utilization: "
                            f"{info['utilization']:.2%}"
                        )
                    break
    finally:
        agent.epsilon = original_epsilon

    print(f"\nTest Results ({num_tests} runs):")
    print(
        "Average Makespan: "
        f"{np.mean(test_makespans):.2f} "
        f"± {np.std(test_makespans):.2f}"
    )
    print(
        "Average Utilization: "
        f"{np.mean(test_utilizations):.2%} "
        f"± {np.std(test_utilizations):.2%}"
    )

    return test_makespans, test_utilizations


def run_policy(env, policy):
    state = env.reset()
    del state

    while True:
        valid_actions = env.get_valid_actions()
        action = policy(env, valid_actions)
        _, _, done, info = env.step(action)

        if done:
            return info


def compare_with_baseline(env):
    print("\nBaseline Comparisons")
    print("=" * 30)

    rr_pointer = {"value": 0}

    def round_robin_policy(local_env, valid_actions):
        for _ in range(local_env.num_jobs):
            candidate = rr_pointer["value"] % local_env.num_jobs
            rr_pointer["value"] += 1
            if candidate in valid_actions:
                return candidate
        return valid_actions[0]

    def spt_policy(local_env, valid_actions):
        def next_processing_time(job_id):
            step = local_env.job_step[job_id]
            machine = local_env.job_sequences[job_id][step]
            return local_env.processing_times[job_id, machine]

        return min(valid_actions, key=next_processing_time)

    rr_info = run_policy(env, round_robin_policy)
    spt_info = run_policy(env, spt_policy)

    print("Round-Robin Baseline:")
    print(f"  Makespan: {rr_info['makespan']:.2f}")
    print(f"  Utilization: {rr_info['utilization']:.2%}")

    print("SPT Baseline:")
    print(f"  Makespan: {spt_info['makespan']:.2f}")
    print(f"  Utilization: {spt_info['utilization']:.2%}")

    return rr_info, spt_info


def plot_training_results(scores, makespans, utilizations):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    window = 50

    def moving_average(values):
        values = np.asarray(values, dtype=float)
        if len(values) < window:
            return np.arange(len(values)), values

        smoothed = np.convolve(
            values,
            np.ones(window) / window,
            mode="valid",
        )
        x = np.arange(window - 1, len(values))
        return x, smoothed

    score_x, score_ma = moving_average(scores)
    makespan_x, makespan_ma = moving_average(makespans)
    util_x, util_ma = moving_average(utilizations)

    ax1.plot(scores, alpha=0.3, label="Raw")
    ax1.plot(score_x, score_ma, label=f"{window}-episode MA")
    ax1.set_title("Training Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(makespans, alpha=0.3, label="Raw")
    ax2.plot(
        makespan_x,
        makespan_ma,
        label=f"{window}-episode MA",
    )
    ax2.set_title("Makespan (Lower = Better)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Makespan")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(utilizations, alpha=0.3, label="Raw")
    ax3.plot(util_x, util_ma, label=f"{window}-episode MA")
    ax3.set_title("Machine Utilization (Higher = Better)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Utilization Rate")
    ax3.legend()
    ax3.grid(True)

    epsilon_curve = [
        max(0.01, 1.0 * (0.995 ** i))
        for i in range(len(scores))
    ]
    ax4.plot(range(len(scores)), epsilon_curve)
    ax4.set_title("Approximate Exploration Rate")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon")
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    print("Manufacturing Job Shop Scheduling with DQN")
    print(
        "Optimizing for makespan, idle time, "
        "and machine utilization"
    )
    print()

    agent, env, scores, makespans, utilizations = (
        train_dqn_scheduler(episodes=500)
    )

    test_makespans, test_utilizations = test_trained_agent(
        agent,
        env,
        num_tests=10,
    )

    rr_info, spt_info = compare_with_baseline(env)

    dqn_avg_makespan = float(np.mean(test_makespans))
    dqn_avg_utilization = float(np.mean(test_utilizations))

    print("\nPerformance Summary")
    print("=" * 40)
    print(
        f"DQN Agent:   Makespan {dqn_avg_makespan:.2f} | "
        f"Utilization {dqn_avg_utilization:.2%}"
    )
    print(
        f"Round Robin: Makespan {rr_info['makespan']:.2f} | "
        f"Utilization {rr_info['utilization']:.2%}"
    )
    print(
        f"SPT:         Makespan {spt_info['makespan']:.2f} | "
        f"Utilization {spt_info['utilization']:.2%}"
    )

    rr_improvement = (
        (rr_info["makespan"] - dqn_avg_makespan)
        / rr_info["makespan"]
        * 100
    )
    spt_improvement = (
        (spt_info["makespan"] - dqn_avg_makespan)
        / spt_info["makespan"]
        * 100
    )

    print("\nManufacturing Improvements")
    print(
        "  Makespan improvement vs Round Robin: "
        f"{rr_improvement:.1f}%"
    )
    print(
        "  Makespan improvement vs SPT: "
        f"{spt_improvement:.1f}%"
    )

    plot_training_results(scores, makespans, utilizations)

    return agent, env


if __name__ == "__main__":
    trained_agent, environment = main()
