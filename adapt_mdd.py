from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
from base_resnet import MyDecay, get_task, get_resnet
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


    X_source = first_blocks.predict(preprocess_input(Xs))
    X_target = first_blocks.predict(preprocess_input(Xt))

    one = OneHotEncoder(sparse=False)
    one.fit(np.array(ys).reshape(-1, 1))

    y_source = one.transform(np.array(ys).reshape(-1, 1))
    y_target = one.transform(np.array(yt).reshape(-1, 1))

    print("X source shape: %s"%str(X_source.shape))
    print("X target shape: %s"%str(X_target.shape))


    mdd.fit(X=X_source[:-1], y=y_source[:-1], Xt=X_target, epochs=epochs, batch_size=batch_size, validation_data=(X_target, y_target))


    acc = mdd.history.history["acc"]
    val_acc = mdd.history.history["val_acc"]
    plt.plot(acc, label="Train acc - final value: %.3f"%acc[-1])
    plt.plot(val_acc, label="Test acc - final value: %.3f"%val_acc[-1])
    plt.legend(); plt.xlabel("Epochs"); plt.ylabel("Acc"); plt.show()


    # Xs_enc = mdd.transform(X_source)
    # Xt_enc = mdd.transform(X_target)

    # np.random.seed(0)
    # X_ = np.concatenate((Xs_enc, Xt_enc))
    # X_tsne = TSNE(2).fit_transform(X_)
    # plt.figure(figsize=(8, 6))
    # plt.plot(X_tsne[:len(X_source), 0], X_tsne[:len(X_source), 1], '.', label="source")
    # plt.plot(X_tsne[len(X_source):, 0], X_tsne[len(X_source):, 1], '.', label="target")
    # plt.legend(fontsize=14)
    # plt.title("Encoded Space tSNE for the MDD model")
    # plt.show()
