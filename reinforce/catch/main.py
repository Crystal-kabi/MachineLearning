import numpy as np
import random
from IPython.display import clear_output
from time import sleep
from datetime import datetime


mode = "trained"


class Environment:

    def __init__(self, size):
        self.action_dict = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1), 4: (0, 0)}
        self.size = size
        self.max_reward = 3 * self.size
        self.penalty = -10
        self.timestep = 0
        self.board, self.reward, self.arrow_pos, self.target_pos = None, None, None, None
        self.reset()

    def renew_reward_array(self, arrow_pos=None, target_pos=None):

        if arrow_pos is None:
            arrow_pos = self.arrow_pos
        if target_pos is None:
            target_pos = self.target_pos

        target_i, target_j = self.pos_decoder(target_pos)
        board = []
        reward = []
        for i in range(self.size):
            row = []
            rwd_row = []
            for j in range(self.size):
                row.append(self.pos_encoder(pos=(i, j)))
                if i == target_i and j == target_j:
                    rwd_row.append(10)
                else:
                    # rwd_row.append(-1)
                    rwd_row.append(2 * self.size - (abs(j - target_j) + abs(i - target_i)))
            board.append(row)
            reward.append(rwd_row)

        reward[target_i][target_j] = self.max_reward
        return board, reward

    def reset(self, size=None):

        if size is None:
            size = self.size
        self.size = size

        init_arrow_i = random.randint(0, self.size - 1)
        init_arrow_j = random.randint(0, self.size - 1)

        init_target_i = random.randint(0, self.size - 1)
        init_target_j = random.randint(0, self.size - 1)

        init_arrow_pos = self.pos_encoder(pos=(init_arrow_i, init_arrow_j))
        init_target_pos = self.pos_encoder(pos=(init_target_i, init_target_j))

        board, reward = self.renew_reward_array(init_arrow_pos, init_target_pos)

        self.board, self.reward, self.arrow_pos, self.target_pos = board, reward, init_arrow_pos, init_target_pos
        self.action_space = self.create_action_space()
        self.state_space = self.create_state_space()
        self.state = self.get_state()
        self.timestep = 0
        return board, reward, init_arrow_pos, init_target_pos

    def pos_encoder(self, pos):
        i = pos[0]
        j = pos[1]
        code = i * self.size + j
        return code

    def pos_decoder(self, code):
        i = int(code / self.size)
        j = code % self.size
        return i, j

    def create_state_space(self):
        state_space = {}
        n = 0
        for i in range(self.size):
            for j in range(self.size):
                arrow_pos = self.pos_encoder((i, j))
                for k in range(self.size):
                    for l in range(self.size):
                        target_pos = self.pos_encoder((k, l))

                        state = (arrow_pos, target_pos)
                        state_space[state] = n

                        n += 1
        return state_space

    def get_state(self):
        state = self.state_space[(self.arrow_pos, self.target_pos)]
        return state

    def get_state_pos(self, state):
        inv_map = {v: k for k, v in self.state_space.items()}
        return inv_map[state]

    def create_action_space(self):
        action_space = []
        old_i, old_j = self.pos_decoder(self.arrow_pos)

        for a in self.action_dict.keys():
            action_i, action_j = self.action_dict[a]

            new_i = old_i + action_i
            new_j = old_j + action_j

            if 0 <= new_i < self.size and 0 <= new_j < self.size:
                action_space.append(a)
        return action_space

    def step(self, action, random_target=False):

        if random_target:
            self.target_pos = random.randint(0, self.size * self.size - 1)

        if action not in self.action_space:
            new_state = self.state
            reward = self.penalty
            done = False

        else:

            old_i, old_j = self.pos_decoder(self.arrow_pos)
            action_i, action_j = self.action_dict[action]

            new_i = old_i + action_i
            new_j = old_j + action_j

            new_pos = self.pos_encoder((new_i, new_j))
            new_state = self.state_space[(new_pos, self.target_pos)]
            reward = self.reward[new_i][new_j]

            self.arrow_pos = new_pos
            self.board, self.reward = self.renew_reward_array()
            self.action_space = self.create_action_space()

            if self.arrow_pos == self.target_pos:
                done = True
            else:
                done = False
        self.timestep += 1

        return new_state, reward, done

    def pretty_board(self, print_board=True):
        arrow_i, arrow_j = self.pos_decoder(code=self.arrow_pos)
        target_i, target_j = self.pos_decoder(code=self.target_pos)
        board_string = ["=" * (2 * self.size + 2) + "\n"]
        for i in range(self.size - 1, -1, -1):
            row_string = ["|"]
            for j in range(self.size):
                if i == arrow_i and j == arrow_j and arrow_i == target_i and arrow_j == target_j:
                    row_string.append("\x1b[42m \x1b[0m")
                elif i == arrow_i and j == arrow_j:
                    row_string.append("\x1b[44m \x1b[0m")
                elif i == target_i and j == target_j:
                    row_string.append("\x1b[41m \x1b[0m")
                else:
                    row_string.append(" ")
                row_string.append(":")
            row_string[-1] = "|\n"
            board_string.append("".join(row_string))
        board_string.append("=" * (2 * self.size + 2) + "\n")
        board_string.append(f"\nStep: {self.timestep}")
        board_string = "".join(board_string)
        if print_board:
            print(board_string)
        return board_string


def random_walk(board):
    boards = [board.pretty_board(print_board=False)]
    step = 0
    done = False

    while not done:
        action_space = board.action_space
        action = random.choice(action_space)

        new_state, reward, done = board.step(action)
        boards.append(board.pretty_board(print_board=False))

        step += 1

    return boards, step, done


def train_Q(init_qtable, environment, epsilon, alpha, gamma, episodes):
    all_epochs = []
    all_penalties = []

    q_table = init_qtable

    start_time = datetime.now()
    whole_progress = 0
    for i in range(episodes + 1):
        environment.reset()
        state = environment.state

        penalties = 0
        reward = 0

        done = False

        while not done:

            action_space = environment.action_space
            if random.uniform(0, 1) < epsilon:
                action = random.choice(action_space)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = environment.step(action)

            old_q = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_q = old_q + alpha * (reward + gamma * next_max - old_q)
            q_table[state, action] = new_q

            if reward == environment.penalty:
                penalties += 1

            state = next_state

        progress = int(i / episodes * 100)
        if progress > whole_progress:
            clear_output(wait=True)
            print(f"\n{progress}% |" + "\x1b[40m \x1b[0m" * progress + "\x1b[39m \x1b[0m" * (100 - progress) + "|\n")
            whole_progress = progress

    end_time = datetime.now()
    print("Training finished")
    print(f"Time taken: {end_time - start_time}")
    return q_table


def walk(init_state, environment, q_table, print_result=True):
    all_steps = [environment.pretty_board(print_board=False)]

    state = init_state

    done = False
    epoch = 0
    penalties = 0
    reward = 0

    while not done:
        action_space = environment.action_space
        action = np.argmax(q_table[state])
        if action not in action_space:
            action = random.choice(action_space)
        state, reward, done = environment.step(action)
        all_steps.append(environment.pretty_board(print_board=False))

        if reward == environment.penalty:
            penalties += 1

        # clear_output(wait=True)
        # environment.pretty_board(print_board=True)
        # sleep(0.1)

    if print_result:
        print(f"No of penalities: {penalties}")
        print(f"No of timesteps: {environment.timestep}")

    return all_steps, epoch, penalties


if mode == "random":
    board = Environment(8)

    boards, step, done = random_walk(board=board)

    for i in boards:
        clear_output(wait=True)
        print(i)
        sleep(1)

else:
    board = Environment(8)
    q_table = np.zeros([len(board.state_space.keys()), len(board.action_dict.keys())])
    new_q_table = train_Q(init_qtable=q_table, environment=board, epsilon=0.5, alpha=0.2, gamma=0.1, episodes=20000)

    board = Environment(8)
    board.pretty_board()

    all_boards, time_step, penny = walk(board.state, board, new_q_table)

    for i in all_boards:
        clear_output(wait=True)
        print(i)
        sleep(1)
    sleep(2)
