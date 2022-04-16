from catch.utils.io import do_not_overwrite, ensure_save_directory_exist, save_pickle
from catch.utils.training_util import train_q_values, apply_qtable
from catch.base.Environment import Environment

from IPython.display import clear_output
from time import sleep


class Model:

    def __init__(self, env_size, metric_class, env_args, metric_args):

        env = Environment(size=env_size, **env_args)
        metric = metric_class(env_size=env.size, **metric_args)
        env.create(metric=metric)

        self.q_table = None
        self.metric = metric
        self.env = env

        self.predict_steps = None

    def train(self, epsilon, alpha, gamma, episodes, init_qtable=None):

        self.q_table = train_q_values(self.env, epsilon, alpha, gamma, episodes, self.metric, init_qtable)

    def predict(self, initial_cfg):

        self.env.reset(metric=self.metric)

        all_steps, epoch, penalty = apply_qtable(initial_cfg, self.env, self.q_table, self.metric, print_result=True)

        self.predict_steps = all_steps

        return all_steps, epoch, penalty

    def show_process(self, overlap=True):

        for i in self.predict_steps:
            if overlap:
                clear_output(wait=True)
            print(i)
            sleep(1)
        sleep(2)

    def save(self, save_path, overwrite=False, skip_same_filename=True):

        if not overwrite:
            if skip_same_filename:
                try:
                    do_not_overwrite(save_path=save_path)
                    ensure_save_directory_exist(save_path=save_path)
                    save_pickle(obj=self, save_path=save_path)
                except FileExistsError:
                    pass
            else:
                do_not_overwrite(save_path=save_path)
                ensure_save_directory_exist(save_path=save_path)
                save_pickle(obj=self, save_path=save_path)
        else:
            ensure_save_directory_exist(save_path=save_path)
            save_pickle(obj=self, save_path=save_path)
