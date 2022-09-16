
import torch
import os

from pytorch_adapt.adapters import DANN, MCD, VADA, CDAN, RTN, ADDA, Aligner, SymNets
from pytorch_adapt.containers import Models, Optimizers, LRSchedulers
from pytorch_adapt.models import Discriminator,  office31C, office31G
from pytorch_adapt.containers import Misc
from pytorch_adapt.layers import RandomizedDotProduct
from pytorch_adapt.layers import MultipleModels, CORALLoss, MMDLoss
from pytorch_adapt.utils import common_functions
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
from pytorch_adapt.validators import AccuracyValidator, IMValidator, ScoreHistory, DiversityValidator, EntropyValidator, MultipleValidators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_source(args, model_name, hp, base_output_dir, results_file, source_domain, target_domain):
    if args.source != None and args.source != source_domain:
        return None, None, None
    if args.target != None and args.target != target_domain:
        return None, None, None

    pair_name = f"{source_domain[0]}2{target_domain[0]}"
    output_dir = os.path.join(base_output_dir, pair_name)
    os.makedirs(output_dir, exist_ok=True)

    print("output dir:", output_dir)

    datasets = get_office31([source_domain], [],
                            folder=args.data_root,
                            return_target_with_labels=True,
                            download=args.download)
    
    dc = DataloaderCreator(batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        )

    weights_root = os.path.join(args.data_root, "weights")

    G = office31G(pretrained=True, model_dir=weights_root).to(device)
    C = office31C(domain=source_domain, pretrained=True,
                  model_dir=weights_root).to(device)


    optimizers = Optimizers((torch.optim.Adam, {"lr": hp.lr}))
    lr_schedulers = LRSchedulers((torch.optim.lr_scheduler.ExponentialLR, {"gamma": hp.gamma}))    

    models = Models({"G": G, "C": C})
    adapter= ClassifierAdapter(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)
    # adapter = get_model(model_name, hp, args.data_root, source_domain)
    
    print("checkpoint dir:", output_dir)
    checkpoint_fn = CheckpointFnCreator(dirname=f"{output_dir}/saved_models", require_empty=False)

    sourceAccuracyValidator = AccuracyValidator()
    val_hooks = [ScoreHistory(sourceAccuracyValidator)]
    
    trainer = Ignite(
        adapter, val_hooks=val_hooks, checkpoint_fn=checkpoint_fn
    )

    early_stopper_kwargs = {"patience": args.patience}

    start_time = datetime.now()

    best_score, best_epoch = trainer.run(
        datasets, dataloader_creator=dc, max_epochs=args.max_epochs, early_stopper_kwargs=early_stopper_kwargs
    )

    end_time = datetime.now()
    training_time = end_time - start_time

    print(f"best_score={best_score}, best_epoch={best_epoch}")

    plt.plot(val_hooks[0].score_history, label='source')
    plt.title("val accuracy")
    plt.legend()
    plt.savefig(f"{output_dir}/val_accuracy.png")
    plt.close('all')

    datasets = get_office31([source_domain], [target_domain], folder=args.data_root, return_target_with_labels=True)
    dc = DataloaderCreator(batch_size=args.batch_size, num_workers=args.num_workers, all_val=True)

    validator = AccuracyValidator(key_map={"src_val": "src_val"})
    src_score = trainer.evaluate_best_model(datasets, validator, dc)
    print("Source acc:", src_score)

    validator = AccuracyValidator(key_map={"target_val_with_labels": "src_val"})
    target_score = trainer.evaluate_best_model(datasets, validator, dc)
    print("Target acc:", target_score)
    print("---------")

    if args.hp_tune:
        with open(results_file, "a") as myfile:
            myfile.write(f"{hp.lr}, {hp.gamma}, {pair_name}, {src_score}, {target_score}, {best_epoch}, {best_score}\n")
    else:
        with open(results_file, "a") as myfile:
            myfile.write(
                f"{pair_name}, {src_score}, {target_score}, {best_epoch}, {best_score}, {training_time.seconds}, ")

    del adapter
    gc.collect()
    torch.cuda.empty_cache()

    return src_score, target_score, best_score
