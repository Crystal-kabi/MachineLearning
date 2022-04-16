import random
import numpy as np
from IPython.display import clear_output
from datetime import datetime


def train_q_values(env, epsilon, alpha, gamma, episodes, metric, init_qtable=None):
    """
    Train the Q table of the environment
    :param env: The environment to be trained in
    :param epsilon: Threshold for not taking random exploration
    :param alpha: Learning rate
    :param gamma: Depreciation factor for considering future rewards
    :param episodes: Number of rounds to train
    :param metric: The metric used to compute rewards
    :param init_qtable: Initial Q table. If not given, initialised as all zeros
    :return: trained Q table
    """

    if init_qtable is None:
        init_qtable = np.zeros([len(env.state_space), len(env.action_dict)])

    q_table = init_qtable

    start_time = datetime.now()
    whole_progress = 0
    for i in range(episodes):

        env.reset(metric=metric)

        state = env.state
        penalty = 0

        done = False
        while not done:

            # get the set of all legal actions in the current state
            action_space = env.action_space

            # choose the action
            if np.random.uniform(0, 1) < epsilon:
                action_index = random.choice(action_space)
            else:
                action_index = np.argmax(q_table[state])

            # take a forward step using the action in the environment
            next_state, reward, done = env.forward_step(action_index=action_index, metric=metric)

            # evaluate the new q value for the action at the current state
            old_q = q_table[state, action_index]
            new_state_q = q_table[next_state]
            q_table[state, action_index] = old_q + alpha * (reward - old_q + gamma * np.max(new_state_q))

            if reward == env.penalty:
                penalty += 1

            # change the state to the new state after taking the action
            state = next_state

        progress = int(i / episodes * 100)
        if progress > whole_progress:
            clear_output(wait=True)
            print(f"Training progress: {progress}% |" + "\x1b[40m \x1b[0m" * progress +
                  "\x1b[39m \x1b[0m" * (100 - progress) + "|", end="\r")
            whole_progress = progress

    end_time = datetime.now()
    print("Training finished")
    print(f"Time taken: {end_time - start_time}")

    return q_table


def apply_qtable(init_state, env, q_table, metric, print_result=True):

    all_steps = [env.pretty_board2d(print_board=False)]

    state = init_state

    done = False
    epoch = 0
    penalty = 0

    while not done:
        action_space = env.action_space
        action_index = np.argmax(q_table[state])
        if action_index not in action_space:
            penalty += 1
            action_index = np.random.choice(action_space)
        state, reward, done = env.forward_step(action_index=action_index, metric=metric)
        all_steps.append(env.pretty_board2d(print_board=False))

        if reward == env.penalty:
            penalty += 1

    if print_result:
        print(f"No of penalties: {penalty}")
        print(f"No of time steps: {env.time_step}")

    return all_steps, epoch, penalty
