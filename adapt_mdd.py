from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
from backbone import MyDecay, get_task, get_resnet, get_input_and_target_for_head
from adapt.feature_based import MDD
from adapt.utils import UpdateLambda


def adapt_mdd_model(Xs, ys, Xt, yt, epochs=50, batch_size=32):
    first_blocks, _, load_resnet50 = get_resnet()

    lr = 0.04
    momentum = 0.9
    alpha = 0.0002

    encoder = load_resnet50()
    task = get_task()

    optimizer_task = SGD(learning_rate=MyDecay(mu_0=lr, alpha=alpha),
                        momentum=momentum, nesterov=True)
    optimizer_enc = SGD(learning_rate=MyDecay(mu_0=lr/10., alpha=alpha),
                        momentum=momentum, nesterov=True)
    optimizer_disc = SGD(learning_rate=MyDecay(mu_0=lr/10., alpha=alpha))


    mdd = MDD(encoder, task,
                loss="categorical_crossentropy",
                metrics=["acc"],
                copy=False,
                lambda_=tf.Variable(0.),
                gamma=2.,
                optimizer=optimizer_task,
                optimizer_enc=optimizer_enc,
                optimizer_disc=optimizer_disc,
                callbacks=[UpdateLambda(lambda_max=0.1)])


    X_source, y_source, X_target, y_target= get_input_and_target_for_head(first_blocks, Xs, ys, Xt, yt)


    mdd.fit(X=X_source[:-1], y=y_source[:-1], Xt=X_target, epochs=epochs, batch_size=batch_size, validation_data=(X_target, y_target))


    acc = mdd.history.history["acc"]
    val_acc = mdd.history.history["val_acc"]
    
    plt.plot(acc, label="Train acc - final value: %.3f"%acc[-1])
    plt.plot(val_acc, label="Test acc - final value: %.3f"%val_acc[-1])
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.savefig('results/mdd.png') 

    return acc, val_acc
