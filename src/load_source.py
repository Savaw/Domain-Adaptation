
import torch
import os

from pytorch_adapt.adapters import DANN, MCD, VADA, CDAN, RTN, ADDA, Aligner, SymNets
from pytorch_adapt.containers import Models, Optimizers, LRSchedulers
from pytorch_adapt.models import Discriminator,  office31C, office31G
from pytorch_adapt.containers import Misc
from pytorch_adapt.containers import LRSchedulers

from classifier_adapter import ClassifierAdapter

from utils import HP, DAModels

import copy

import matplotlib.pyplot as plt
import torch
import os
import gc
from datetime import datetime

from pytorch_adapt.datasets import DataloaderCreator, get_office31
from pytorch_adapt.frameworks.ignite import CheckpointFnCreator, Ignite
from pytorch_adapt.validators import AccuracyValidator, IMValidator, ScoreHistory 
from pytorch_adapt.frameworks.ignite import (
        CheckpointFnCreator,
        IgniteValHookWrapper,
        checkpoint_utils,
    )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_source_trainer(checkpoint_dir):

    G = office31G(pretrained=False).to(device)
    C = office31C(pretrained=False).to(device)


    optimizers = Optimizers((torch.optim.Adam, {"lr": 1e-4}))
    lr_schedulers = LRSchedulers((torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.99}))    

    models = Models({"G": G, "C": C})
    adapter= ClassifierAdapter(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)


    checkpoint_fn = CheckpointFnCreator(dirname=checkpoint_dir, require_empty=False)

    sourceAccuracyValidator = AccuracyValidator()
    val_hooks = [ScoreHistory(sourceAccuracyValidator)]

    new_trainer = Ignite(
        adapter, val_hooks=val_hooks, checkpoint_fn=checkpoint_fn, device=device
    )

    objs = [
            {
                "engine": new_trainer.trainer,
                **checkpoint_utils.adapter_to_dict(new_trainer.adapter),
            }
        ]

    for to_load in objs:
        checkpoint_fn.load_best_checkpoint(to_load)

    return new_trainer
