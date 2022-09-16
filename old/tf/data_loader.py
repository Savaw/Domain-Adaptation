import os
import tensorflow as tf
import numpy as np 
from sklearn.preprocessing import OneHotEncoder


DOMAINS = ["amazon", "dslr", "webcam"]


def load_data(source_domain,
              target_domain,
              validation_split=0.2,
              seed=123,
              image_size=(150, 150),
              batch_size=32,
              base_path="../datasets/office31"):
    """
    source_domains: List of domain(s) used for training and evaluating
    target_domain: List of domain(s) used for testing
    base_path: Base directory of office31 dataset. The structure of this directory should be as follows: 
                {domain_name}/{class_name}/{image_name}.jpg
                e.g. amazon/bike_helmet/frame_0003.jpg
    """

    if source_domain not in DOMAINS or target_domain not in DOMAINS:
        raise ValueError("Invalid domains")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(base_path, source_domain),
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(base_path, source_domain),
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size)

    target_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(base_path, target_domain),
        seed=seed,
        image_size=image_size,
        batch_size=batch_size)

    return train_ds, val_ds, target_ds

def get_input_and_labels_from_batch_ds(dataset):
    xy = [(x, y) for x, y in dataset]
    X = np.concatenate([x for x, y in xy], axis=0)
    Y = np.concatenate([y for x, y in xy], axis=0)

    one = OneHotEncoder(sparse_output=False)
    one.fit(np.array(Y).reshape(-1, 1))

    Y = one.transform(np.array(Y).reshape(-1, 1))

    return X, Y
