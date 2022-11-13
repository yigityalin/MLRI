from pathlib import Path
from typing import Callable, Optional

from tensorflow import keras
import tensorflow as tf

from utils import config


def check_model_path(model_path: str | Path):
    """
    Checks whether the model_path is suitable.
    :param model_path: the path to which model will be saved
    :return: None
    """
    model_path = Path(model_path)
    if model_path == config.MODELS_DIR:
        raise RuntimeError(f'model_path cannot be the same as {config.MODELS_DIR.as_posix()}')
    if not model_path.is_relative_to(config.MODELS_DIR):
        raise RuntimeError(f'model_path must be in {config.MODELS_DIR.as_posix()}')
    if model_path.exists():
        raise RuntimeError('model_path already exists.')
    if model_path.suffix != '.h5':
        raise RuntimeError('The models must be saved as .h5 files.')


def create_model(*, preprocessing_layers: keras.layers.Layer | None = None,
                 base_model: Callable[..., keras.layers.Layer] | None = None,
                 top_layers: keras.layers.Layer | None,
                 include_top: bool = False,
                 pooling: None | str = 'avg',
                 input_shape: Optional[tuple[int, int, int]] = None,
                 classes: int = 4,
                 classifier_activation: None | str | Callable = 'softmax',
                 compile_model: bool = True,
                 loss: list[keras.losses.Loss, ...] = None,
                 optimizer: keras.optimizers.Optimizer = None,
                 metrics: list[str, ...] | list[keras.metrics.Metric, ...] = None,
                 learning_rate: float = 5e-3,
                 ) -> keras.Model:
    """
    Combines multiple models/layers and creates a model.
    Note that a keras.Model is also a keras.layers.Layer.
    Stacks the inputs in the following order:
        preprocessing_layers
        base_model
        top_layers
    :param preprocessing_layers: the layer that preprocesses the input image
    :param base_model: the function that creates the base model for transfer learning.
    :param top_layers: the layer that is added on top of the whole network.
    :param include_top: whether to include the fully-connected layer at the top of the network.
    :param pooling: pooling mode for feature extraction. Must be None or str ('avg' or 'max')
    :param input_shape: the shape of the input
    :param classes: Optional number of classes to classify images into,
                    only to be specified if include_top is True, and if no weights argument is specified.
    :param classifier_activation: A string or callable. The activation function to use on the "top" layer.
                                  Ignored unless include_top=True. Set classifier_activation=None to
                                  return the logits of the "top" layer. When loading pretrained weights,
                                  classifier_activation can only be None or "softmax".
    :param compile_model: whether to compile the model
    :param loss: the model loss. Ignored unless compile is True.
    :param optimizer: the model optimizer. Ignored unless compile is True.
    :param metrics: the model metrics. Ignored unless compile is True.
    :param learning_rate: the learning rate with which the Nadam optimizer is initialized.
                          Ignored unless compile is True and optimizer is not specified.
    :return: the model
    """
    module_list = list()
    if preprocessing_layers:
        module_list.append(preprocessing_layers)
    if base_model:
        base = base_model(include_top=include_top, pooling=pooling,
                          input_shape=input_shape, classes=classes,
                          classifier_activation=classifier_activation)
        for layer in base.layers:
            layer.trainable = False
        module_list.append(base)
    if top_layers:
        module_list.append(top_layers)

    if len(module_list) == 0:
        raise ValueError('You must specify at least one of the following parameters: '
                         'preprocessing_layers, base_model, top_layers')

    model = keras.Sequential(module_list)
    if compile_model:
        if loss is None:
            loss = [keras.losses.SparseCategoricalCrossentropy()]
        if optimizer is None:
            optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
        if metrics is None:
            metrics = [keras.metrics.SparseCategoricalAccuracy()]

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


class RandomHue(keras.layers.Layer):
    """
    A preprocessing layer that adjusts the hue of RGB images
    by a random factor chosen in the interval [-max_delta, max_delta].
    Active only during training
    """
    def __init__(self, max_delta):
        """
        Initializes the layer
        :param max_delta: the maximum adjustment of the hue. Must be in the interval [0, 0.5]
        """
        super(RandomHue, self).__init__()
        self.max_delta = max_delta

    def call(self, inputs, training=None):
        if training:
            inputs = tf.image.random_hue(inputs, self.max_delta)
        return inputs

    def get_config(self):
        config = super(RandomHue, self).get_config()
        config.update({'max_delta': self.max_delta})
        return config


class RandomSaturation(keras.layers.Layer):
    """
    A preprocessing layer that adjust the saturation of RGB images
    by a random factor chosen in the interval [lower, upper).
    Active only during training.
    """
    def __init__(self, lower, upper):
        """
        Initializes the layer
        :param lower: the lower bound of the interval from which the adjustment parameter is randomly chosen
        :param upper: the upper bound of the interval from which the adjustment parameter is randomly chosen
        """
        super(RandomSaturation, self).__init__()
        self.lower = lower
        self.upper = upper

    def call(self, inputs, training=None):
        if training:
            inputs = tf.image.random_saturation(inputs, self.lower, self.upper)
        return inputs

    def get_config(self):
        config = super(RandomSaturation, self).get_config()
        config.update({
            "lower": self.lower,
            "upper": self.upper,
        })
        return config
