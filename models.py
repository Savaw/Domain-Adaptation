
import torch
import os

from pytorch_adapt.adapters import DANN, MCD, VADA, CDAN, RTN, ADDA, Aligner
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name, hp: HP, data_root, source_domain):
    weights_root = os.path.join(data_root, "weights")

    G = office31G(pretrained=True, model_dir=weights_root).to(device)
    C = office31C(domain=source_domain, pretrained=True,
                  model_dir=weights_root).to(device)
    D = Discriminator(in_size=2048, h=1024).to(device)

    optimizers = Optimizers((torch.optim.Adam, {"lr": hp.lr}))
    lr_schedulers = LRSchedulers((torch.optim.lr_scheduler.ExponentialLR, {"gamma": hp.gamma}))
    
    def del_model_params():
        del G
        del C
        del D

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

        def del_model_params():
            del G
            del C
            del D
            del C1
            del C_combined

    elif model_name ==  DAModels.MMD:
        models = Models({"G": G, "C": C})
        hook_kwargs = {"loss_fn": MMDLoss}
        adapter= Aligner(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers, hook_kwargs=hook_kwargs)

    elif model_name == DAModels.CORAL:
        models = Models({"G": G, "C": C})
        hook_kwargs = {"loss_fn": CORALLoss}
        adapter= Aligner(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers, hook_kwargs=hook_kwargs)

    elif model_name ==  DAModels.Source:
        models = Models({"G": G, "C": C})
        adapter= ClassifierAdapter(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)

    return adapter, del_model_params

