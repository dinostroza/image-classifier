import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from PIL import Image


def process_image(image):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def argtopmax(array, top_k):
    return array.argsort()[-top_k:][::-1]


def get_classnames(path):
    with open(path, 'r') as f:
        class_names = json.load(f)
    return class_names


def get_image_batch(image):
    return np.expand_dims(image, axis=0)


def get_image(path):
    image = Image.open(path)
    return np.asarray(image)


def load_model(model_filepath):
    h5_model = tf.keras.models.load_model(
        model_filepath,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )

    return h5_model


def get_probs(image, model):
    # get a batch with the formated image
    single_image_batch = get_image_batch(image)
    # get array probabilities for batch
    batch_probs = model.predict(single_image_batch, verbose=0)

    return batch_probs[0]
