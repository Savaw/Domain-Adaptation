import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from data_loader import load_data, get_input_and_labels_from_batch_ds

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

epochs=50
batch_size=32

train_ds, val_ds, target_ds = load_data("amazon", "webcam", image_size=(224,224))
Xs, ys = get_input_and_labels_from_batch_ds(train_ds)
Xv, yv = get_input_and_labels_from_batch_ds(val_ds)
Xt, yt = get_input_and_labels_from_batch_ds(target_ds)

from pretrained_resnet import pretrained_resnet_model

# pretrained_resnet_model(Xs, ys, Xt, yt, epochs=epochs, batch_size=batch_size)


from adapt_mdd import adapt_mdd_model

adapt_mdd_model(Xs, ys, Xt, yt, epochs=epochs, batch_size=batch_size)
