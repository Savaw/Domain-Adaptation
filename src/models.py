
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
from load_source import get_source_trainer

from utils import HP, DAModels

import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name, hp: HP, source_checkpoint_dir):
    source_trainer = get_source_trainer(source_checkpoint_dir)
    
    G = copy.deepcopy(source_trainer.adapter.models["G"])
    C = copy.deepcopy(source_trainer.adapter.models["C"])
    D = Discriminator(in_size=2048, h=1024).to(device)

    # if True:
    #     G.fc = C.net[:6]
    #     C.net = C.net[6:]
    #     D = Discriminator(in_size=128, h=256).to(device)

    optimizers = Optimizers((torch.optim.Adam, {"lr": hp.lr}))
    lr_schedulers = LRSchedulers((torch.optim.lr_scheduler.ExponentialLR, {"gamma": hp.gamma}))
    # lr_schedulers = LRSchedulers((torch.optim.lr_scheduler.MultiStepLR, {"milestones": [2, 5, 10, 20, 40], "gamma": hp.gamma}))
    

    if model_name == DAModels.DANN:
        models = Models({"G": G, "C": C, "D": D})
        adapter = DANN(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)

    elif model_name ==  DAModels.CDAN:
        models = Models({"G": G, "C": C, "D": D})
        misc = Misc({"feature_combiner": RandomizedDotProduct([2048, 31], 2048)})
        adapter = CDAN(models=models, misc=misc, optimizers=optimizers, lr_schedulers=lr_schedulers)

    elif model_name ==  DAModels.MCD:
        C1 = common_functions.reinit(copy.deepcopy(C))
        C_combined = MultipleModels(C, C1)
        models = Models({"G": G, "C": C_combined})
        adapter= MCD(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)

    elif model_name ==  DAModels.SYMNET:
        C1 = common_functions.reinit(copy.deepcopy(C))
        C_combined = MultipleModels(C, C1)
        models = Models({"G": G, "C": C_combined})
        adapter= SymNets(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)

    elif model_name ==  DAModels.MMD:
        models = Models({"G": G, "C": C})
        hook_kwargs = {"loss_fn": MMDLoss()}
        adapter= Aligner(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers, hook_kwargs=hook_kwargs)

    elif model_name == DAModels.CORAL:
        models = Models({"G": G, "C": C})
        hook_kwargs = {"loss_fn": CORALLoss()}
        adapter= Aligner(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers, hook_kwargs=hook_kwargs)

    elif model_name ==  DAModels.SOURCE:
        models = Models({"G": G, "C": C})
        adapter= ClassifierAdapter(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)


    return adapter

