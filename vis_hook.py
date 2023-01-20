
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import umap
import os
import gc
from datetime import datetime

from pytorch_adapt.adapters import DANN
from pytorch_adapt.containers import Models, Optimizers, LRSchedulers
from pytorch_adapt.datasets import DataloaderCreator, get_office31
from pytorch_adapt.frameworks.ignite import CheckpointFnCreator, Ignite
from pytorch_adapt.models import Discriminator, office31C, office31G
from pytorch_adapt.validators import AccuracyValidator, IMValidator, ScoreHistory

class VizHook:
    def __init__(self, **kwargs):
        self.required_data = ["src_val",
                              "target_val", "target_val_with_labels"]
        self.kwargs = kwargs

    def __call__(self, epoch, src_val, target_val, target_val_with_labels, **kwargs):

        accuracy_validator = AccuracyValidator()
        accuracy = accuracy_validator.compute_score(src_val=src_val)
        print("src_val accuracy:", accuracy)
        accuracy_validator = AccuracyValidator()
        accuracy = accuracy_validator.compute_score(
            src_val=target_val_with_labels)
        print("target_val accuracy:", accuracy)

        if epoch >= 1 and epoch % kwargs.get("frequency", 5) != 0:
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
