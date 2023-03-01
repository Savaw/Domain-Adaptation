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
from pytorch_adapt.adapters.base_adapter import BaseGCAdapter
from pytorch_adapt.adapters.utils import with_opt
from pytorch_adapt.hooks import ClassifierHook
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

PATIENCE = 5
EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 2
TRIAL_COUNT = 5

logging.basicConfig()
logging.getLogger("pytorch-adapt").setLevel(logging.WARNING)

class ClassifierAdapter(BaseGCAdapter):
    """
    Wraps [AlignerPlusCHook][pytorch_adapt.hooks.AlignerPlusCHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["G", "C"]```|
    """

    def init_hook(self, hook_kwargs):
        opts = with_opt(list(self.optimizers.keys()))
        self.hook = self.hook_cls(opts, **hook_kwargs)

    @property
    def hook_cls(self):
        return ClassifierHook

root='/content/drive/MyDrive/Shared with Sabas/Bsc/'
# root="datasets/pytorch-adapt/"

data_root = os.path.join(root,'data')

batch_size=BATCH_SIZE
num_workers=NUM_WORKERS

device = torch.device("cuda")
model_dir = os.path.join(data_root, "weights")

DATASET_PAIRS = [("amazon", ["webcam", "dslr"]),
                    ("webcam", ["dslr", "amazon"]),
                    ("dslr", ["amazon", "webcam"])
                    ]

MODEL_NAME = "base"
model_name = MODEL_NAME

pass_next= 0
pass_trial = 0
for trial_number in range(10, 10 + TRIAL_COUNT):
    if pass_trial:
        pass_trial -= 1
        continue

    base_output_dir = f"{root}/results/vishook/{MODEL_NAME}/{trial_number}"
    os.makedirs(base_output_dir, exist_ok=True)

    d = datetime.now()
    results_file = f"{base_output_dir}/{d.strftime('%Y%m%d-%H:%M:%S')}.txt"

    with open(results_file, "w") as myfile:
        myfile.write("pair, source_acc, target_acc, best_epoch, time\n")

    for source_domain, target_domains in DATASET_PAIRS:
      datasets = get_office31([source_domain], [], folder=data_root)
      dc = DataloaderCreator(batch_size=batch_size, 
                              num_workers=num_workers, 
                              )
      dataloaders = dc(**datasets)        

      G = office31G(pretrained=True, model_dir=model_dir)
      C = office31C(domain=source_domain, pretrained=True, model_dir=model_dir)

      optimizers = Optimizers((torch.optim.Adam, {"lr": 0.0005}))
      lr_schedulers = LRSchedulers((torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.99}))

      if model_name == "base":
        models = Models({"G": G, "C": C})
        adapter= ClassifierAdapter(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)
                  

      checkpoint_fn = CheckpointFnCreator(dirname="saved_models", require_empty=False)
      val_hooks = [ScoreHistory(AccuracyValidator())]
      
      trainer = Ignite(
          adapter, val_hooks=val_hooks, checkpoint_fn=checkpoint_fn, 
      )

      early_stopper_kwargs = {"patience": PATIENCE}

      start_time = datetime.now()

      best_score, best_epoch = trainer.run(
          datasets, dataloader_creator=dc, max_epochs=EPOCHS, early_stopper_kwargs=early_stopper_kwargs
      )

      end_time = datetime.now()
      training_time = end_time - start_time

      for target_domain in target_domains:
        if pass_next:
          pass_next -= 1
          continue

        pair_name = f"{source_domain[0]}2{target_domain[0]}"
        output_dir = os.path.join(base_output_dir, pair_name)
        os.makedirs(output_dir, exist_ok=True)

        print("output dir:", output_dir)

        # print(f"best_score={best_score}, best_epoch={best_epoch}, training_time={training_time.seconds}")

        plt.plot(val_hooks[0].score_history, label='source')
        plt.title("val accuracy")
        plt.legend()
        plt.savefig(f"{output_dir}/val_accuracy.png") 
        plt.close('all')
        

        datasets = get_office31([source_domain], [target_domain], folder=data_root, return_target_with_labels=True)
        dc = DataloaderCreator(batch_size=64, num_workers=2, all_val=True)

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
