import numpy as np
from catch.utils.envrionment_util import pos2index, index2pos
from catch.standard_objects.env_op import standard_action, standard_size, standard_penalty


class Environment:

    def __init__(self, size=standard_size, action_dict=None, penalty=standard_penalty):
        """

        :param size: dimension of the environment in the order of (d1, d2, ..., dn)
        :param action_dict: a dictionary. values are functions to be applied on positions in the environment
        :param penalty: penalty value for when an illegal step is taken
        """

        if action_dict is None:
            action_dict = standard_action

        self.action_dict = action_dict
        if isinstance(size, int):
            size = (size, )
        self.size = size

        self.time_step = 0

        self.reward = None
        self.penalty = penalty

        self.current_pos = None
        self.target_pos = None

        self.state_space = None
        self.state = None

        self.action_space = None

    def create(self, **kwargs):
        """
        Create the environment
        :param kwargs: keyword arguments to be supplied to the reset method
        :return: None
        """

        self.reset(**kwargs)

    def update_reward_array(self, target_pos_index, metric):
        """
        Create rewards for each next possible position
        :param target_pos_index: position index of the target in the environment
        :param metric: metric to compute the reward
        :return: (prod(self.size), ) ndarray containing the reward
        """

        size_array = np.asarray(self.size)

        # get the coordinates of each position
        env_config_index = np.asarray(range(size_array.prod()))
        env_config_coordinate = np.apply_along_axis(lambda x: index2pos(self.size, x), -1,
                                                    env_config_index.reshape(-1, 1))

        # compute reward for each position
        target_pos_coordinate = env_config_coordinate[target_pos_index]
        reward = metric.score(env_config_coord=env_config_coordinate, target_coord=target_pos_coordinate)

        return reward

    def reset(self, init_pos=None, init_target=None, metric=None):
        """
        reset the state of the environment
        :param init_pos: position after reset
        :param init_target: target position after reset
        :param metric: metric to compute reward after reset
        :return: None
        """

        size_array = np.asarray(self.size)

        if init_pos is None:
            init_position_index = np.random.randint(size_array.prod())
        else:
            init_position_index = pos2index(env_size=self.size, position=init_pos)

        if init_target is None:
            init_target_index = np.random.randint(size_array.prod())
        else:
            init_target_index = pos2index(env_size=self.size, position=init_target)

        self.reward = self.update_reward_array(target_pos_index=init_target_index, metric=metric)
        self.current_pos = init_position_index
        self.target_pos = init_target_index

        self.state_space = self.create_state_space()
        self.action_space = self.create_action_space()

        self.state = self.get_state()
        self.time_step = 0

    def create_state_space(self):
        """
        Create the state space for the environment
        :return: state space dictionary, keys are (current position index, target position index),
        values are the state index
        """

        size_array = np.asarray(self.size)

        state_space = {}

        possible_pos_index = np.asarray(range(size_array.prod()))

        state_index = 0
        for curr_pos in possible_pos_index:
            for target_pos in possible_pos_index:
                state = (curr_pos, target_pos)
                state_space[state] = state_index
                state_index += 1

        return state_space

    def create_action_space(self, position_index=None):
        """
        Create the action space for the given position
        :param position_index: Index of the position
        :return: 1 x number of legal actions ndarray
        """

        size_array = np.asarray(self.size)

        if position_index is None:
            position_index = self.current_pos

        action_space = []
        old_position_coordinate = index2pos(env_size=self.size, index=position_index)
        for action_index, action in self.action_dict.items():

            new_coordinate = np.asarray(action(old_position_coordinate))

            if np.all(np.logical_and(new_coordinate >= 0, new_coordinate < size_array)):
                action_space.append(action_index)

        action_space = np.asarray(action_space)
        self.action_space = action_space

        return action_space

    def get_state(self):
        state = self.state_space[(self.current_pos, self.target_pos)]
        return state

    def get_state_pos(self, state):
        inv_map = {v: k for k, v in self.state_space.items()}
        return inv_map[state]

    def forward_step(self, action_index, metric):
        """
        Take a forward step in the environment
        :param action_index: The index of action to be taken
        :param metric: metric to compute the reward for the action
        :return: (state after action, reward for the action, whether the target is caught)
        """

        done = False

        if action_index not in self.action_space:

            new_state = self.state
            reward = self.penalty

        else:

            action = self.action_dict[action_index]

            old_position_coord = index2pos(self.size, self.current_pos)
            new_position_coord = action(old_position_coord)
            new_position_index = pos2index(self.size, new_position_coord)
            new_state = self.state_space[(new_position_index, self.target_pos)]
            reward = self.reward[new_position_index]

            self.current_pos = pos2index(self.size, new_position_coord)
            self.reward = self.update_reward_array(target_pos_index=self.target_pos, metric=metric)
            self.action_space = self.create_action_space(position_index=self.current_pos)

            if self.current_pos == self.target_pos:
                done = True

        self.time_step += 1

        return new_state, reward, done

    def pretty_board2d(self, print_board=True):
        """
        Create a string representation of a 2-d board
        :param print_board: When True, print the board
        :return: string of 2-d board
        """
        current_x, current_y = index2pos(self.size, index=self.current_pos)
        target_x, target_y = index2pos(self.size, index=self.target_pos)
        board_string = ["=" * (2 * self.size[0] + 2) + "\n"]
        for row in range(self.size[1]):
            row_string = ["|"]
            for col in range(self.size[0]):
                if row == current_y and col == current_x and current_y == target_y and current_x == target_x:
                    row_string.append("\x1b[42m \x1b[0m")
                elif row == current_y and col == current_x:
                    row_string.append("\x1b[44m \x1b[0m")
                elif row == target_y and col == target_x:
                    row_string.append("\x1b[41m \x1b[0m")
                else:
                    row_string.append(" ")
                row_string.append(":")
            row_string[-1] = "|\n"
            board_string.append("".join(row_string))
        board_string.append("=" * (2 * self.size[0] + 2) + "\n")
        board_string.append(f"\nStep: {self.time_step}")
        board_string = "".join(board_string)
        if print_board:
            print(board_string)
        return board_string
