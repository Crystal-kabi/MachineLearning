from catch.base.Model import Model
from catch.base.Environment import standard_action, standard_penalty
from catch.CustomMetric import L1Distance

clf = Model(env_size=(4, 6), metric_class=L1Distance,
            env_args={"action_dict": standard_action, "penalty": standard_penalty},
            metric_args={"score_at_target": 10})
clf.train(epsilon=0.5, alpha=0.2, gamma=0.1, episodes=20000)
all_env_progress, time_step, penalty = clf.predict(clf.env.state)
clf.show_process()
