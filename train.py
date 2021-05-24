
import matplotlib.pyplot as plt
import torch
import os
import gc
from datetime import datetime

from pytorch_adapt.datasets import DataloaderCreator, get_office31
from pytorch_adapt.frameworks.ignite import CheckpointFnCreator, Ignite
from pytorch_adapt.validators import AccuracyValidator, IMValidator, ScoreHistory

from models import get_model

from vis_hook import VizHook

def train(args, model_name, hp, base_output_dir, results_file, source_domain, target_domain):
    if args.source != None and args.source != source_domain:
        return None, None, None
    if args.target != None and args.target != target_domain:
        return None, None, None

    pair_name = f"{source_domain[0]}2{target_domain[0]}"
    output_dir = os.path.join(base_output_dir, pair_name)
    os.makedirs(output_dir, exist_ok=True)

    print("output dir:", output_dir)

    datasets = get_office31([source_domain], [target_domain],
                            folder=args.data_root,
                            return_target_with_labels=True,
                            download=args.download)
    dc = DataloaderCreator(batch_size=args.batch_size,
                           num_workers=args.num_workers,
                           train_names=["train"],
                           val_names=["src_train", "target_train", "src_val", "target_val",
                                        "target_train_with_labels", "target_val_with_labels"])

    adapter = get_model(model_name, hp, args.data_root, source_domain)
    
    checkpoint_fn = CheckpointFnCreator(dirname=f"{output_dir}/saved_models", require_empty=False)
    scoreValidator = ScoreHistory(IMValidator())
    sourceAccuracyValidator = AccuracyValidator()
    targetAccuracyValidator = AccuracyValidator(key_map={"target_val_with_labels": "src_val"})

    
    val_hooks = [ScoreHistory(sourceAccuracyValidator), 
                ScoreHistory(targetAccuracyValidator),
                VizHook(output_dir=output_dir, frequency=args.vishook_frequency)]
    
    trainer = Ignite(
        adapter, validator=scoreValidator, val_hooks=val_hooks, checkpoint_fn=checkpoint_fn
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
    plt.plot(val_hooks[1].score_history, label='target')
    plt.title("val accuracy")
    plt.legend()
    plt.savefig(f"{output_dir}/val_accuracy.png")
    plt.close('all')

    plt.plot(scoreValidator.score_history)
    plt.title("score_history")
    plt.savefig(f"{output_dir}/score_history.png")
    plt.close('all')

    src_score = val_hooks[0].score_history[-1]
    print("Source acc:", src_score)

    target_score = val_hooks[1].score_history[-1]
    print("Target acc:", target_score)

    if args.hp_tune:
        with open(results_file, "a") as myfile:
            myfile.write(f"{hp.lr}, {hp.gamma}, {pair_name}, {target_score}, {best_epoch}, {best_score}\n")
    else:
        with open(results_file, "a") as myfile:
            myfile.write(
                f"{pair_name}, {src_score}, {target_score}, {best_epoch}, {training_time.seconds}\n")

    del adapter
    gc.collect()
    torch.cuda.empty_cache()

    return src_score, target_score, best_score
