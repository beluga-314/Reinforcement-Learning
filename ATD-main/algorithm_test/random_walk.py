
import sys

sys.path.append(".")

from atd import TDAgent, SVDATDAgent, DiagonalizedSVDATDAgent, PlainATDAgent ,tLSTDAgent, Backend
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

N = 7
v_true = Backend.linspace(1, N, N - 1) / (N - 1)
v = None
rng = np.random.default_rng()


def evaluate(w):
    observation_count = 0
    absolute_error = 0
    for observation in Backend.eye(N)[1:N]:  # Generate the one-hot inputs.
        absolute_error += abs((w @ observation - v_true[observation_count]) / v_true[observation_count])
        observation_count += 1
    return absolute_error / observation_count


def play_game(agent, episodes=100, iterations=100):
    global v

    records = []

    for _ in trange(iterations):
        record = []
        agent.reinit()
        agent.w = Backend.zeros(N)
        t = 0

        for i in range(episodes):
            pos = (N - 1) / 2
            observation = Backend.eye(N)[int((N - 1) / 2)]
            agent.reset()
            record.append(evaluate(agent.w))

            while True:
                pos += rng.choice((-1, 1))
                next_observation = Backend.eye(N)[int(pos)]

                if pos == N - 1:
                    agent.learn(observation, next_observation, 1, 0, t)
                    break

                if pos == 0:
                    agent.learn(observation, next_observation, 0, 0, t)
                    break

                agent.learn(observation, next_observation, 0, 1, t)
                observation = next_observation
                t += 1

        records.append(record)
    return Backend.mean(Backend.create_matrix_func(records), 0)


plt.figure(dpi=120, figsize=(8, 6))

plt.plot(play_game(agent=TDAgent(lr=0.1, lambd=0, observation_space_n=7, action_space_n=2),
                   iterations=10, episodes=100), label="TD(0.5), $\\alpha=0.1$")
plt.plot(play_game(agent=DiagonalizedSVDATDAgent(k=30, eta=1e-4, lambd=0, observation_space_n=7, action_space_n=2),
                   iterations=10, episodes=100),
         label="DiagonalizedSVDATD(0.5), $\\alpha=\\frac{1}{1+t}$, \n$\\eta=1\\times10^{-4}$, $r=30$")
plt.plot(play_game(agent=DiagonalizedSVDATDAgent(k=30, eta=1e-4, lambd=0.3, observation_space_n=7,
                                                 action_space_n=2, trace_update_mode='emphatic'),
                   iterations=10, episodes=100),
         label="Emphatic DiagonalizedSVDATD(0.5), $\\alpha=\\frac{1}{1+t}$, \n$\\eta=1\\times10^{-4}$, $r=30$")
plt.plot(play_game(agent=tLSTDAgent(k=10, eta=1e-4, lambd=0, observation_space_n=7,
                                                 action_space_n=2),
                   iterations=10),
         label="tLSTDAgent(0), $\\alpha=\\frac{1}{1+t}$, \n$\\eta=1\\times10^{-4}$, $r=30$")
plt.plot(play_game(agent=PlainATDAgent(eta=1e-4, lambd=0, observation_space_n=7, action_space_n=2),
                   iterations=10, episodes=100),
         label="PlainATD(0.5), $\\alpha=\\frac{1}{1+t}$, $\\eta=1\\times10^{-4}$")
plt.legend()
plt.title("Random Walk")
plt.xlabel("Episode")
plt.ylabel("Percentage Error")
plt.xlim(0, 100)
plt.ylim(0.2, 1)
plt.savefig("./../figures/random_walk.png", format="png")
