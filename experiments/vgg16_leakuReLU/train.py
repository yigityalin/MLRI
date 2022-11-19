from pathlib import Path
import sys

root_dir = Path().resolve().parent.parent.parent.as_posix()
if root_dir not in sys.path:
    sys.path.append(root_dir)
del root_dir

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf

from utils import config, datasets, models

MODEL_NAME = 'VGG16'
EPOCHS = 200

train_dataset, validation_dataset, test_dataset = datasets.load_dataset()

preprocessing = keras.Sequential([
    layers.Rescaling(1. / 255)
])

top = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same',
                  activation=keras.layers.LeakyReLU(alpha=0.07),
                  kernel_initializer='he_normal'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                  activation=keras.layers.LeakyReLU(alpha=0.07),
                  kernel_initializer='he_normal'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.05),
                 kernel_initializer='he_normal'),
    layers.Dense(4, kernel_initializer='he_normal',
                 activation=keras.activations.softmax)
])

model = models.create_model(
    model_name=MODEL_NAME,
    preprocessing_layers=preprocessing,
    base_model=None,
    top_layers=top,
)

history = models.fit_model(
    model,
    train_data=train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
