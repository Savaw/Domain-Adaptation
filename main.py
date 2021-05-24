import argparse
import logging
import os
from datetime import datetime
from train import train
from utils import HP, DAModels

logging.basicConfig()
logging.getLogger("pytorch-adapt").setLevel(logging.WARNING)


DATASET_PAIRS = [("amazon", "webcam"), ("amazon", "dslr"),
                ("webcam", "amazon"), ("webcam", "dslr"),
                ("dslr", "amazon"), ("dslr", "webcam")]


def run_experiment_on_model(args, model_name):
    for trial_number in range(args.initial_trial, args.initial_trial + args.trials_count):
        base_output_dir = f"{args.results_root}/results/vishook/{model_name}/{trial_number}"
        os.makedirs(base_output_dir, exist_ok=True)

        d = datetime.now()
        results_file = f"{base_output_dir}/{d.strftime('%Y%m%d-%H:%M:%S')}.txt"

        with open(results_file, "w") as myfile:
            myfile.write("pair, source_acc, target_acc, best_epoch, time\n")

        hp = HP(lr=args.lr, gamma=args.gamma)

        for source_domain, target_domain in DATASET_PAIRS:
            logging.info(f"Running experiment on model {model_name} trail {trial_number}/{args.trials_count} pair {source_domain}2{target_domain}")
            train(args, model_name, hp, base_output_dir, results_file, source_domain, target_domain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', default=60, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--trials_count', default=3, type=int)
    parser.add_argument('--initial_trial', default=0, type=int)
    parser.add_argument('--download', default=False, type=bool)
    parser.add_argument('--root', default="./")
    parser.add_argument('--data_root', default="./datasets/pytorch-adapt/")
    parser.add_argument('--results_root', default="./results/vishook/")
    parser.add_argument('--model_names', default=["DANN"], nargs='+')
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    

    args = parser.parse_args()

    print(args)

    for model_name in args.model_names:
        try:
            model_enum = DAModels(model_name)
        except ValueError:
            logging.warning(f"Model {model_name} not found. skipping...")
            continue

        run_experiment_on_model(args, model_enum)
