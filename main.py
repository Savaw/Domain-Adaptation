import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import umap
from tqdm import tqdm
import os

from pytorch_adapt.adapters import DANN
from pytorch_adapt.containers import Models, Optimizers, LRSchedulers
from pytorch_adapt.datasets import DataloaderCreator, get_mnist_mnistm, get_office31
from pytorch_adapt.frameworks.ignite import CheckpointFnCreator, Ignite
from pytorch_adapt.models import Discriminator, mnistC, mnistG, office31C, office31G
from pytorch_adapt.validators import AccuracyValidator, IMValidator, ScoreHistory

from pprint import pprint


logging.basicConfig()
# logging.getLogger("pytorch-adapt").setLevel(logging.INFO)
logging.getLogger("pytorch-adapt").setLevel(logging.WARNING)


class VizHook:
    def __init__(self):
        self.required_data = ["src_val", "target_val", "target_val_with_labels"]

    def __call__(self, epoch, src_val, target_val, target_val_with_labels, **kwargs):

        accuracy_validator = AccuracyValidator()
        accuracy = accuracy_validator.compute_score(src_val=src_val)
        print("src_val accuracy:", accuracy)
        accuracy_validator = AccuracyValidator()
        accuracy = accuracy_validator.compute_score(src_val=target_val_with_labels)
        print("target_val accuracy:", accuracy)

        if epoch % 5 != 0:
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
        plt.savefig(f"results/vishook/dann/val_{epoch}.png") 
        plt.close('all')



root="datasets/pytorch-adapt/"
batch_size=32
num_workers=2

datasets = get_office31(["amazon"], ["webcam"], folder=root, return_target_with_labels=True)
dc = DataloaderCreator(batch_size=batch_size, 
                        num_workers=num_workers, 
                        train_names=["train"],
                        val_names=["src_train", "target_train", "src_val", "target_val", "target_train_with_labels", "target_val_with_labels"])
dataloaders = dc(**datasets)




device = torch.device("cuda")
weights_root = os.path.join(root, "weights")
trained_domain = "amazon"

G = office31G(pretrained=True, model_dir=weights_root).to(device)
C = office31C(domain=trained_domain, pretrained=True, model_dir=weights_root).to(device)
D = Discriminator(in_size=2048, h=1024).to(device)

models = Models({"G": G, "C": C, "D": D})

optimizers = Optimizers((torch.optim.Adam, {"lr": 0.0005}))
lr_schedulers = LRSchedulers((torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.99}))

adapter = DANN(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)
checkpoint_fn = CheckpointFnCreator(dirname="results/saved_models", require_empty=False)
validator = ScoreHistory(IMValidator())
tarAccuracyValidator = AccuracyValidator(key_map={"target_val_with_labels":"src_val"})
val_hooks = [ScoreHistory(AccuracyValidator()), ScoreHistory(tarAccuracyValidator), VizHook()]
trainer = Ignite(
    adapter, validator=validator, val_hooks=val_hooks, checkpoint_fn=checkpoint_fn
)


early_stopper_kwargs = {"patience": 15}

best_score, best_epoch = trainer.run(
    datasets, dataloader_creator=dc, max_epochs=100, early_stopper_kwargs=early_stopper_kwargs
)

print(f"best_score={best_score}, best_epoch={best_epoch}")


plt.plot(validator.score_history[1:])
plt.title("score_history")
plt.show()

plt.plot(val_hooks[0].score_history[1:], label='source')
plt.plot(val_hooks[1].score_history[1:], label='target')
plt.title("val accuracy")
plt.legend()
plt.savefig(f"results/vishook/dann/val_acc.png") 
plt.close('all')


validator = AccuracyValidator(key_map={"src_val": "src_val"})
score = trainer.evaluate_best_model(datasets, validator, dc)
print("Source acc:", score)

validator = AccuracyValidator(key_map={"target_val_with_labels": "src_val"})
score = trainer.evaluate_best_model(datasets, validator, dc)
print("Target acc:", score)
