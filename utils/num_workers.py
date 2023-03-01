
from pytorch_adapt.datasets import DataloaderCreator, get_office31
from pytorch_adapt.frameworks.ignite import CheckpointFnCreator, Ignite
from pytorch_adapt.validators import AccuracyValidator, IMValidator, ScoreHistory, DiversityValidator, EntropyValidator, MultipleValidators

from time import time
import multiprocessing as mp

data_root = "../datasets/pytorch-adapt/"
batch_size = 32
for num_workers in range(2, mp.cpu_count(), 2):  
    datasets = get_office31(["amazon"], ["webcam"],
                                folder=data_root,
                                return_target_with_labels=True,
                                download=False)
        
    dc = DataloaderCreator(batch_size=batch_size,
                        num_workers=num_workers,
                        train_names=["train"],
                        val_names=["src_train", "target_train", "src_val", "target_val",
                                        "target_train_with_labels", "target_val_with_labels"])
    dataloaders = dc(**datasets)

    train_loader = dataloaders["train"]
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
