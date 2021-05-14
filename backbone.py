import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

def get_resnet():
    # if you want to download weights, remove weights param in ResNet40 and remove this line
    WEIGHTS_PATH = 'model-weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    resnet50 = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling="avg", weights=WEIGHTS_PATH, classes=31)

    first_layer = resnet50.get_layer('conv5_block2_out')
    inputs = Input(first_layer.output_shape[1:])

    for layer in resnet50.layers[resnet50.layers.index(first_layer)+1:]:
        if layer.name == "conv5_block3_1_conv":
            x = layer(inputs)
        elif layer.name == "conv5_block3_add":
            x = layer([inputs, x])
        else:
            x = layer(x)

    first_blocks = Model(resnet50.input, first_layer.output)
    last_block = Model(inputs, x)


    def load_resnet50(path="model-weights/resnet50_last_block.hdf5"):
        model = load_model(path, compile=False)
        for i in range(len(model.layers)):
            if model.layers[i].__class__.__name__ == "BatchNormalization":
                model.layers[i].trainable = False
        return model

    # last_block.summary()
    last_block.save("model-weights/resnet50_last_block.hdf5")

    return first_blocks, last_block, load_resnet50


def get_task(dropout=0.5, max_norm=0.5):
    model = Sequential()
    model.add(Dense(1024, activation="relu",
                kernel_constraint=MaxNorm(max_norm),
                bias_constraint=MaxNorm(max_norm)))
    model.add(Dropout(dropout))
    model.add(Dense(1024, activation="relu",
                kernel_constraint=MaxNorm(max_norm),
                bias_constraint=MaxNorm(max_norm)))
    model.add(Dropout(dropout))
    model.add(Dense(31, activation="softmax",
                kernel_constraint=MaxNorm(max_norm),
                bias_constraint=MaxNorm(max_norm)))
    return model


class MyDecay(LearningRateSchedule):
    def __init__(self, max_steps=1000, mu_0=0.01, alpha=10, beta=0.75):
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta
        self.max_steps = float(max_steps)

    def __call__(self, step):
        p = step / self.max_steps
        return self.mu_0 / (1+self.alpha * p)**self.beta




def get_input_and_target_for_head(first_blocks, X_arr):
    new_X_arr = []
    for x in X_arr:
        new_x = first_blocks.predict(preprocess_input(x))
        new_X_arr.append(new_x)

    return new_X_arr
    