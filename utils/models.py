from pathlib import Path
from typing import Callable, Optional

from tensorflow import keras
import tensorflow as tf

from utils import config


def create_model(*, model_name: str,
                 preprocessing_layers: keras.layers.Layer | None = None,
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
    :param model_name: the name of the model
    :param preprocessing_layers: the layer that preprocesses the input image
    :param base_model: the function that creates the base model for transfer learning.
    :param top_layers: the layer that is added on top of the whole network.
    :param include_top: whether to include the fully-connected layer at the top of the network.
    :param pooling: pooling mode for feature extraction. Must be None or str ('avg' or 'max')
    :param input_shape: the shape of the input
    :param classes: Optional number of classes to classify images into, only to be specified if include_top is True
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

    model = keras.Sequential(module_list, name=model_name)
    if compile_model:
        if loss is None:
            loss = [keras.losses.SparseCategoricalCrossentropy()]
        if optimizer is None:
            optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
        if metrics is None:
            metrics = [keras.metrics.SparseCategoricalAccuracy()]

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def fit_model(model: keras.Model,
              model_path: None | str | Path = None,
              log_dir: None | str | Path = None,
              history_path: None | str | Path = None,
              train_data=None,
              validation_data=None,
              epochs=1,
              verbose="auto",
              callbacks=None,
              shuffle=True,
              class_weight=None,
              initial_epoch=0,
              steps_per_epoch=None,
              validation_freq=1,
              max_queue_size=10,
              ) -> keras.callbacks.History:
    """

    :param model: the model to be trained
    :param model_path: the path to which model will be saved
    :param log_dir: the path to which tensorboard logs will be saved
    :param history_path: the path to which the history callback will be saved
    :param train_data: the training data

    the rest of the parameter information can be found in the documentation of the fit method of keras.Model.
    :return: the history callback
    """
    if model_path is None:
        model_path = config.get_default_model_path(model.name, epochs)
    if log_dir is None:
        log_dir = config.get_default_tensorboard_logs_dir(model.name, epochs)
    if history_path is None:
        history_path = config.get_default_history_path(model.name, epochs)

    config.check_model_path(model_path)
    config.check_logs_path(log_dir)
    config.check_logs_path(history_path)

    if callbacks is None:
        callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir)]

    history = model.fit(train_data,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks,
                        validation_data=validation_data,
                        shuffle=shuffle,
                        class_weight=class_weight,
                        initial_epoch=initial_epoch,
                        steps_per_epoch=steps_per_epoch,
                        validation_freq=validation_freq,
                        max_queue_size=max_queue_size)

    model.save(model_path)
    config.save_history(history, history_path)
    return history


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
