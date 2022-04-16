from catch.base.Metric import Metric
from catch.utils.envrionment_util import pos2index
import numpy as np


class L1Distance(Metric):

    def __init__(self, score_at_target, env_size=None):

        super(L1Distance, self).__init__(score_at_target=score_at_target)

        self.env_size = env_size

    def score(self, env_config_coord, target_coord, **kwargs):

        size_array = np.asarray(self.env_size)

        reward = np.sum(size_array - np.abs((env_config_coord - target_coord)), axis=-1)
        reward[pos2index(self.env_size, target_coord)] = self.score_at_target

        return reward
