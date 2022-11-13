from tensorflow import keras
import numpy as np
import tensorflow as tf

from . import config


def load_dataset(
        directory=config.DATA_DIR, valid_split: float = 0.1, test_split: float = 0.1, batch_size: int = 32,
        image_size: tuple[int, int] = (128, 128), shuffle_buffer_size: int = 8, seed: int = 42,
        prefetch_buffer_size: int = tf.data.AUTOTUNE, cache: bool = True, return_class_names: bool = False
) -> tuple[tf.data.Dataset, ...] | tuple[tuple[tf.data.Dataset, ...], list[str, ...]]:
    """
    Loads and splits the image dataset
    :param directory: the data directory
    :param valid_split: the fraction of the data reserved for validation dataset
    :param test_split: the fraction of the data reserved for test dataset
    :param batch_size: the size of batches of data
    :param image_size: the size to resize images to after they are read from disk
    :param prefetch_buffer_size: the maximum number of elements that will be buffered when prefetching
    :param shuffle_buffer_size: the number of elements from this dataset from which the new dataset will sample.
    :param seed: the random seed
    :param cache: whether to cache the dataset
    :param return_class_names: whether to return the class names along with the datasets
    :return: the datasets
    """
    if valid_split < 0 or test_split < 0:
        raise ValueError('valid_split and test_split must be positive.')
    if valid_split + test_split > 1:
        raise ValueError('valid_split + test_split must be less than 1.')

    dataset = keras.utils.image_dataset_from_directory(directory, image_size=image_size, batch_size=batch_size,
                                                       shuffle=shuffle_buffer_size > 0, seed=seed)
    class_names = dataset.class_names
    if cache:
        dataset = dataset.cache()
    cardinality = dataset.cardinality().numpy()
    validation_size, test_size = cardinality * valid_split, cardinality * test_split

    datasets = list()
    if validation_size > 0:
        validation_dataset = dataset.take(validation_size)
        dataset = dataset.skip(validation_size)
        datasets += validation_dataset,

    if test_size > 0:
        test_dataset = dataset.take(test_size)
        dataset = dataset.skip(test_size)
        datasets += test_dataset,

    datasets.insert(0, dataset)
    if shuffle_buffer_size:
        datasets = [dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True, seed=seed)
                    for dataset in datasets]
    if prefetch_buffer_size:
        datasets = [dataset.prefetch(buffer_size=prefetch_buffer_size) for dataset in datasets]

    if return_class_names:
        return tuple(datasets), class_names
    else:
        return tuple(datasets)


def get_class_distribution(dataset: tf.data.Dataset) -> np.ndarray:
    """
    returns the class distribution a dataset
    :param dataset: a tf.data.Dataset
    :return: the number of instances belonging to each class
    """
    labels = np.concatenate([labels for _, labels in dataset.as_numpy_iterator()])
    return np.unique(labels, return_counts=True)[1]
