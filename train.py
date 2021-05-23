import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import umap
from tqdm import tqdm
import os
import gc
from datetime import datetime
import copy

from pytorch_adapt.adapters import MCD
from pytorch_adapt.containers import Models, Optimizers, LRSchedulers
from pytorch_adapt.datasets import DataloaderCreator, get_office31
from pytorch_adapt.frameworks.ignite import CheckpointFnCreator, Ignite
from pytorch_adapt.models import Discriminator, office31C, office31G
from pytorch_adapt.validators import AccuracyValidator, IMValidator, ScoreHistory
from pytorch_adapt.containers import Misc
from pytorch_adapt.layers import RandomizedDotProduct
from pytorch_adapt.layers import MultipleModels
from pytorch_adapt.utils import common_functions 
from pytorch_adapt.containers import LRSchedulers

from pprint import pprint


logging.basicConfig()
logging.getLogger("pytorch-adapt").setLevel(logging.WARNING)


class VizHook:
    def __init__(self, **kwargs):
        self.required_data = ["src_val", "target_val", "target_val_with_labels"]
        self.kwargs = kwargs

    def __call__(self, epoch, src_val, target_val, target_val_with_labels, **kwargs):

        accuracy_validator = AccuracyValidator()
        accuracy = accuracy_validator.compute_score(src_val=src_val)
        print("src_val accuracy:", accuracy)
        accuracy_validator = AccuracyValidator()
        accuracy = accuracy_validator.compute_score(src_val=target_val_with_labels)
        print("target_val accuracy:", accuracy)

        if epoch >= 1 and epoch % 5 != 0:
            return

        features = [src_val["features"], target_val["features"]]
        domain = [src_val["domain"], target_val["domain"]]
        features = torch.cat(features, dim=0).cpu().numpy()
        domain = torch.cat(domain, dim=0).cpu().numpy()
        emb = umap.UMAP().fit_transform(features)

        df = pd.DataFrame(emb).assign(domain=domain)
        df["domain"] = df["domain"].replace({0: "Source", 1: "Target"})
        sns.set_theme(style="white", rc={"figure.figsize": (8, 6)})
        sns.scatterplot(data=df, x=0, y=1, hue="domain", s=10)
        plt.savefig(f"{self.kwargs['output_dir']}/val_{epoch}.png") 
        plt.close('all')


def train(adapter, source_domain, target_domain, results_file, data_root, base_output_dir, batch_size, num_workers, patience):
    pair_name = f"{source_domain[0]}2{target_domain[0]}"
    output_dir = os.path.join(base_output_dir, pair_name)
    os.makedirs(output_dir, exist_ok=True)

    print("output dir:", output_dir)

    datasets = get_office31([source_domain], [target_domain], folder=data_root, return_target_with_labels=True)
    dc = DataloaderCreator(batch_size=batch_size, 
                            num_workers=num_workers, 
                            train_names=["train"],
                            val_names=["src_train", "target_train", "src_val", "target_val", "target_train_with_labels", "target_val_with_labels"])
    dataloaders = dc(**datasets)

    

    device = torch.device("cuda")
    weights_root = os.path.join(data_root, "weights")

    G = office31G(pretrained=True, model_dir=weights_root).to(device)
    C0 = office31C(domain=source_domain, pretrained=True, model_dir=weights_root).to(device)
    C1 = common_functions.reinit(copy.deepcopy(C0))
    C = MultipleModels(C0, C1)

    models = Models({"G": G, "C": C})

    optimizers = Optimizers((torch.optim.Adam, {"lr": 0.0005}))
    lr_schedulers = LRSchedulers((torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.99}))

    adapter= MCD(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)

    checkpoint_fn = CheckpointFnCreator(dirname=os.path.join(output_dir, "saved_models"), require_empty=False)
    scoreValidator = ScoreHistory(IMValidator())
    srcAccuracyValidator = AccuracyValidator()
    tarAccuracyValidator = AccuracyValidator(key_map={"target_val_with_labels":"src_val"})
    val_hooks = [ScoreHistory(srcAccuracyValidator), ScoreHistory(tarAccuracyValidator), VizHook(output_dir=output_dir)]
    trainer = Ignite(
        adapter, validator=scoreValidator, val_hooks=val_hooks, checkpoint_fn=checkpoint_fn
    )

    start_time = datetime.now()

    early_stopper_kwargs = {"patience": patience}

    best_score, best_epoch = trainer.run(
        datasets, dataloader_creator=dc, max_epochs=epochs, early_stopper_kwargs=early_stopper_kwargs
    )

    end_time = datetime.now()
    training_time = end_time - start_time
    

    print(f"best_score={best_score}, best_epoch={best_epoch}")


    plt.plot(val_hooks[0].score_history, label='source')
    plt.plot(val_hooks[1].score_history, label='target')
    plt.title("val accuracy")
    plt.legend()
    plt.savefig(f"{output_dir}/val_accuracy.png") 
    plt.close('all')

    plt.plot(validator.score_history)
    plt.title("score_history")
    plt.savefig(f"{output_dir}/score_history.png") 
    plt.close('all')

    validator = AccuracyValidator(key_map={"src_val": "src_val"})
    src_score = trainer.evaluate_best_model(datasets, validator, dc)
    print("Source acc:", src_score)

    validator = AccuracyValidator(key_map={"target_val_with_labels": "src_val"})
    target_score = trainer.evaluate_best_model(datasets, validator, dc)
    print("Target acc:", target_score)

    with open(results_file, "a") as myfile:
        myfile.write(f"{pair_name}, {src_score}, {target_score}, {best_epoch}, {training_time.seconds}\n")

    del trainer
    del G
    del C
    gc.collect()
    torch.cuda.empty_cache()
