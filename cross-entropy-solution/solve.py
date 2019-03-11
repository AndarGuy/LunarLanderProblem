import gym.wrappers
import numpy as np
from sklearn.neural_network import MLPClassifier

env = gym.make("LunarLander-v2")
state = env.reset()

N_ACTIONS = 4
N_STATE = 8
BATCH_SIZE = 100
N_EPISODES = 200
PERCENTILE = 80

agent = MLPClassifier(warm_start=True, random_state=0, hidden_layer_sizes=(64, 64))
agent.fit([state] * N_ACTIONS, [0, 1, 2, 3])


def get_batch(n_frames=500):
    states = []
    actions = []
    reward = 0

    s = env.reset()
    for frame in range(n_frames):
        a = np.random.choice(N_ACTIONS, p=agent.predict_proba([s])[0])
        states.append(s)
        actions.append(a)

        s, r, done, _ = env.step(a)
        reward += r

        if done:
            break

    return states, actions, reward


def show():
    s = env.reset()
    for frame in range(200):
        a = np.random.choice(N_ACTIONS, p=agent.predict_proba([s])[0])
        s, r, done, _ = env.step(a)
        env.render()
        if done:
            break


for episode in range(N_EPISODES):

    batch_states, batch_actions, batch_rewards = np.array([get_batch() for _ in range(BATCH_SIZE)]).T

    threshold = np.percentile(batch_rewards, PERCENTILE)
    need_states = np.concatenate(
        [batch_state for i, batch_state in enumerate(batch_states) if batch_rewards[i] >= threshold])
    need_actions = np.concatenate(
        [batch_action for i, batch_action in enumerate(batch_actions) if batch_rewards[i] >= threshold])

    agent.fit(need_states, need_actions)

    if episode % 10 == 0:
        print('\rEpisode {}\tAverage Rewards: {:.2f}'.format(episode, np.mean(batch_rewards)))
        show()

    print('\rEpisode {}\tAverage Rewards: {:.2f}'.format(episode, np.mean(batch_rewards)), end="")
