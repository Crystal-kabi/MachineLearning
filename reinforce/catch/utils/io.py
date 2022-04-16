import pickle
import os


def do_not_overwrite(save_path):

    if os.path.isfile(save_path):
        raise FileExistsError("File already exist. {}".format(save_path))


def ensure_save_directory_exist(save_path):

    save_directory = os.path.dirname(os.path.abspath(save_path))
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)


def save_pickle(obj, save_path):

    with open(save_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
