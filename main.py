import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from data_loader import load_data, get_input_and_labels_from_batch_ds

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


epochs=200
batch_size=64

train_ds, val_ds, target_ds = load_data("amazon", "webcam", image_size=(224,224))
Xs, ys = get_input_and_labels_from_batch_ds(train_ds)
Xv, yv = get_input_and_labels_from_batch_ds(val_ds)
Xt, yt = get_input_and_labels_from_batch_ds(target_ds)

from pretrained_resnet import pretrained_resnet_model

# pretrained_resnet_model(Xs, ys, Xt, yt, epochs=epochs, batch_size=batch_size)

from train import train_model

MODEL_NAMES = [
    # "base", 
    "DeepCORAL", 
    # "MCD", 
    # "MDD", 
    # "WDGRL", 
    # "CDAN", 
    # "CCSA", 
    # "DANN", 
    # "ADDA"
    ]

d = datetime.now()
results_file = f"results/e{epochs}_b{batch_size}_{d.strftime('%Y%m%d-%H:%M:%S')}.txt"

with open(results_file, "w") as myfile:
    myfile.write("model_name, acc, disc_acc, val_acc, source_scoure, val_score, target_score\n")


for model_name in MODEL_NAMES:
    train_model(model_name, Xs, ys, Xv, yv, Xt, yt, epochs=epochs, batch_size=batch_size, results_file=results_file)
