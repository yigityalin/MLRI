from pathlib import Path
import pickle
import os

import keras.callbacks

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT_DIR / 'data'


def get_models_dir() -> Path:
    """
    Gets the valid models directory considering the project structure
    :return: the models directory
    """
    models_dir = Path().resolve().parent / 'models'
    if not models_dir.exists():
        models_dir.mkdir(exist_ok=True)
    return models_dir


def get_logs_dir() -> Path:
    """
    Gets the valid logs directory considering the project structure
    :return: the logs directory
    """
    logs_dir = Path().resolve().parent / 'logs'
    if not logs_dir.exists():
        logs_dir.mkdir(exist_ok=True)
    return logs_dir


def get_default_model_path(model_name: str, total_epochs: int) -> Path:
    """
    returns the default model path
    :param model_name: the name of the model
    :param total_epochs: the total number of epochs the model will be trained for
    :return: the model path
    """
    return get_models_dir() / f'model_{model_name}-total_epochs_{total_epochs}.h5'


def get_default_tensorboard_logs_dir(model_name: str, total_epochs: int) -> Path:
    """
    returns the default logs directory
    :param model_name: the name of the model
    :param total_epochs: the total number of epochs the model will be trained for
    :return: the logs directory
    """
    return get_logs_dir() / f'logs_tensorboard_{model_name}-total_epochs_{total_epochs}'


def get_default_checkpoint_dir() -> Path:
    """
    returns the default model checkpoint directory
    :return: the model directory
    """
    checkpoint_dir = Path().resolve().parent / 'model_checkpoints'
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


def get_default_history_path(model_name: str, total_epochs: int) -> Path:
    """
    returns the default history path
    :param model_name: the name of the model
    :param total_epochs: the total number of epochs the model will be trained for
    :return: the history path
    """
    return get_logs_dir() / f'history_{model_name}-total_epochs_{total_epochs}.pkl'


def save_history(history: keras.callbacks.History, path: str | Path):
    """
    pickles and saves a keras.callbacks.History instance
    :param history: the keras.callbacks.History instance
    :param path: the path to which the history will be saved
    :return: None
    """
    path = Path(path)
    with open(path, 'wb') as pkl_history:
        pickle.dump(history.history, pkl_history)


def load_history(path: str | Path) -> keras.callbacks.History:
    """
    Loads a pickled keras.callbacks.History instance
    :param path: the path for pickled keras.callbacks.History instance
    :return: the unpickled keras.callbacks.History instance
    """
    path = Path(path)
    with open(path, "rb") as pkl_history:
        history = pickle.load(pkl_history)
    return history


def check_path_validity(path: str | Path, true_dir: str | Path) -> None:
    """
    Checks whether a path is valid for project structure
    :param path: the path to be checked
    :param true_dir: the true directory to which the path will be relative
    :return: None
    """
    path = Path(path)
    true_dir = Path(true_dir)
    if path == true_dir:
        raise RuntimeError(f'model_path cannot be the same as {true_dir.as_posix()}')
    if not path.is_relative_to(true_dir):
        raise RuntimeError(f'model_path must be in {true_dir.as_posix()}')
    if path.exists():
        raise RuntimeError('model_path already exists.')


def check_model_path(model_path: str | Path) -> None:
    """
    Checks whether the model_dir is suitable.
    Wrapper around the check_path_validity function.
    :param model_path: the path to which model will be saved
    :return: None
    """
    model_path = Path(model_path)
    model_dir = get_models_dir()
    check_path_validity(model_path, model_dir)
    if model_path.suffix != '.h5':
        raise RuntimeError('The models must be saved as .h5 files.')


def check_logs_path(logs_dir: str | Path) -> None:
    """
    Checks whether the logs_dir is suitable.
    Wrapper around the check_path_validity function.
    :param logs_dir: the directory to which the logs will be saved
    :return: None
    """
    logs_dir = Path(logs_dir)
    true_logs_dir = get_logs_dir()
    check_path_validity(logs_dir, true_logs_dir)
