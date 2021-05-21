
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import SGD
from adapt.parameter_based import FineTuning
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
from backbone import MyDecay, get_task, get_resnet, get_input_and_target_for_head
from sklearn.manifold import TSNE
from adapt.utils import UpdateLambda
from adapt.feature_based import DeepCORAL,MCD,MDD, WDGRL ,CDAN ,CCSA ,DANN,ADDA

def train_model(model_name, Xs, ys, Xv, yv, Xt, yt, epochs=50, batch_size=32, results_file=None):
    first_blocks, _, load_resnet50 = get_resnet()

    print("Getting input...")
    Xs, Xv, Xt = get_input_and_target_for_head(first_blocks, [Xs, Xv, Xt])

    print("Getting model...")
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

    target_data_available = True

    if model_name == "base":
        model = FineTuning(encoder=encoder,
                            task=task,
                            optimizer=optimizer_task,
                            optimizer_enc=optimizer_enc,
                            loss="categorical_crossentropy",
                            metrics=["acc"],
                            copy=False,
                            pretrain=True,
                            pretrain__epochs=5)
        target_data_available = False

    elif model_name == "MDD":
        model = MDD(encoder, task,
                loss="categorical_crossentropy",
                metrics=["acc"],
                copy=False,
                gamma=2.,
                lambda_=tf.Variable(0.),
                optimizer=optimizer_task,
                optimizer_enc=optimizer_enc,
                optimizer_disc=optimizer_disc,
                callbacks=[UpdateLambda(lambda_max=0.1)])
    elif model_name =="DeepCORAL":
        model = DeepCORAL(encoder, task,
                    loss="categorical_crossentropy",
                    metrics=["acc"],
                    copy=False,
                    lambda_=tf.Variable(0.),
                    optimizer=optimizer_task,
                    optimizer_enc=optimizer_enc,
                    optimizer_disc=optimizer_disc,
                    callbacks=[UpdateLambda(lambda_max=0.1)])
    elif model_name =="MCD": 
        model = MCD(encoder, task,
                    loss="categorical_crossentropy",
                    metrics=["acc"],
                    copy=False,
                    optimizer=optimizer_task,
                    optimizer_enc=optimizer_enc,
                    optimizer_disc=optimizer_disc)
    elif model_name == "WDGRL":
        model = WDGRL(encoder, task,
                    loss="categorical_crossentropy",
                    metrics=["acc"],
                    copy=False,
                    optimizer=optimizer_task,
                    optimizer_enc=optimizer_enc,
                    optimizer_disc=optimizer_disc,
                    )
    elif model_name == "CDAN":
        model = CDAN(encoder, task,
                    loss="categorical_crossentropy",
                    metrics=["acc"],
                    copy=False,
                    optimizer=optimizer_task,
                    optimizer_enc=optimizer_enc,
                    optimizer_disc=optimizer_disc,
                    )
    elif model_name == "CCSA":
        model = CCSA(encoder, task,
                    loss="categorical_crossentropy",
                    metrics=["acc"],
                    copy=False,
                    optimizer=optimizer_task,
                    optimizer_enc=optimizer_enc,
                    optimizer_disc=optimizer_disc,
                    )
    elif model_name == "DANN":
        model = DANN(encoder, task,
                    loss="categorical_crossentropy",
                    metrics=["acc"],
                    copy=False,
                    optimizer=optimizer_task,
                    optimizer_enc=optimizer_enc,
                    optimizer_disc=optimizer_disc,
                    )
    elif model_name == "ADDA":
        model = ADDA(encoder, task,
                    loss="categorical_crossentropy",
                    metrics=["acc"],
                    copy=False,
                    optimizer=optimizer_task,
                    optimizer_enc=optimizer_enc,
                    optimizer_disc=optimizer_disc,
                    )
    
    
    print(f"Training {model.name}...")
    if target_data_available:
        model.fit(Xs, ys, Xt=Xt, yt=yt, epochs=epochs, batch_size=batch_size, validation_data=(Xv, yv))
    else:
        model.fit(Xs, ys, epochs=epochs, batch_size=batch_size, validation_data=(Xv, yv))


    acc = model.history.history.get("acc", None) 
    disc_acc = model.history.history.get("disc_acc", None)
    val_acc = model.history.history["val_acc"]
    
    source_score = model.score(Xs, ys)
    val_score = model.score(Xv, yv)
    target_score = model.score(Xt, yt)


    result = f"{model_name}, {acc[-1] if acc else None}, {disc_acc[-1] if disc_acc else None}, {val_acc[-1] if val_acc else None}, {source_score}, {val_score}, {target_score}\n"
    print("####", result)

    print(f"Storing results...")

    if results_file:
        with open(results_file, "a") as myfile:
            myfile.write(result)

    plt.plot(acc, label="Train acc - final value: %.3f"%acc[-1])
    plt.plot(val_acc, label="Test acc - final value: %.3f"%val_acc[-1])
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.savefig(f"results/{model.name}.png") 
    plt.close('all')
    