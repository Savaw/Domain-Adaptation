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

from pytorch_adapt.adapters import DANN, MCD, VADA, CDAN, RTN, ADDA, Aligner
from pytorch_adapt.containers import Models, Optimizers, LRSchedulers
from pytorch_adapt.datasets import DataloaderCreator, get_mnist_mnistm, get_office31
from pytorch_adapt.frameworks.ignite import CheckpointFnCreator, Ignite
from pytorch_adapt.models import Discriminator, mnistC, mnistG, office31C, office31G
from pytorch_adapt.validators import AccuracyValidator, IMValidator, ScoreHistory
from pytorch_adapt.containers import Misc
from pytorch_adapt.layers import RandomizedDotProduct
from pytorch_adapt.layers import MultipleModels
from pytorch_adapt.utils import common_functions 
from pytorch_adapt.containers import LRSchedulers
import copy

from pprint import pprint

PATIENCE = 10
EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 2
TRIAL_COUNT = 5

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


root='/content/drive/MyDrive/Shared with Sabas/Bsc/'
# root="datasets/pytorch-adapt/"

data_root = os.path.join(root,'data')

batch_size=BATCH_SIZE
num_workers=NUM_WORKERS

device = torch.device("cuda")
model_dir = os.path.join(data_root, "weights")

DATASET_PAIRS = [("amazon", "webcam"), ("amazon", "dslr"),
                    ("webcam", "dslr"), ("webcam", "amazon"),
                    ("dslr", "amazon"), ("dslr", "webcam")]

MODEL_NAME = "cdan"
model_name = MODEL_NAME

pass_next= 4
pass_trial = 1
for trial_number in range(TRIAL_COUNT):
    if pass_trial:
        pass_trial -= 1
        continue

    base_output_dir = f"{root}/results/vishook/{MODEL_NAME}/{trial_number}"
    os.makedirs(base_output_dir, exist_ok=True)

    d = datetime.now()
    results_file = f"{base_output_dir}/{d.strftime('%Y%m%d-%H:%M:%S')}.txt"

    with open(results_file, "w") as myfile:
        myfile.write("pair, source_acc, target_acc, best_epoch, time\n")

    for source_domain, target_domain in DATASET_PAIRS:
        if pass_next:
          pass_next -= 1
          continue
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

        G = office31G(pretrained=True, model_dir=model_dir)
        C = office31C(domain=source_domain, pretrained=True, model_dir=model_dir)
        D = Discriminator(in_size=2048, h=1024)

        optimizers = Optimizers((torch.optim.Adam, {"lr": 0.0005}))
        lr_schedulers = LRSchedulers((torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.99}))

        if model_name == "cdan":
          models = Models({"G": G, "C": C, "D": D})
          misc = Misc({"feature_combiner": RandomizedDotProduct([2048, 31], 2048)})
          adapter = CDAN(models=models, misc=misc, optimizers=optimizers, lr_schedulers=lr_schedulers)

        elif model_name == "aligner":
          models = Models({"G": G, "C": C})
          adapter= Aligner(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)
          

        checkpoint_fn = CheckpointFnCreator(dirname="saved_models", require_empty=False)
        validator = ScoreHistory(IMValidator())
        tarAccuracyValidator = AccuracyValidator(key_map={"target_val_with_labels":"src_val"})
        val_hooks = [ScoreHistory(AccuracyValidator()), ScoreHistory(tarAccuracyValidator), VizHook(output_dir=output_dir)]
        
        trainer = Ignite(
            adapter, validator=validator, val_hooks=val_hooks, checkpoint_fn=checkpoint_fn, 
        )

        early_stopper_kwargs = {"patience": PATIENCE}

        start_time = datetime.now()

        best_score, best_epoch = trainer.run(
            datasets, dataloader_creator=dc, max_epochs=EPOCHS, early_stopper_kwargs=early_stopper_kwargs
        )

        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"best_score={best_score}, best_epoch={best_epoch}, training_time={training_time.seconds}")

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
        del D
        gc.collect()
        torch.cuda.empty_cache()
