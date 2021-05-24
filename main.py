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
        base_output_dir = f"{args.results_root}/{model_name}/{trial_number}"
        os.makedirs(base_output_dir, exist_ok=True)        

        if not args.hp_tune:
            hp = HP(lr=args.lr, gamma=args.gamma)

            d = datetime.now()
            results_file = f"{base_output_dir}/e{args.max_epochs}_p{args.patience}_lr{hp.lr}_g{hp.gamma}_{d.strftime('%Y%m%d-%H:%M:%S')}.txt"
            with open(results_file, "w") as myfile:
                myfile.write("pair, source_acc, target_acc, best_epoch, time\n")

            for source_domain, target_domain in DATASET_PAIRS:
                train(args, model_name, hp, base_output_dir, results_file, source_domain, target_domain)
        
        else:
            hp_values = {
                0.01: [0.8, 0.5],
                0.001: [1, 0.9, 0.8],
                0.0001: [1, 0.9, 0.99]
            }

            d = datetime.now()
            hp_file = f"{base_output_dir}/hp_e{args.max_epochs}_p{args.patience}_{d.strftime('%Y%m%d-%H:%M:%S')}.txt"
            with open(hp_file, "w") as myfile:
                myfile.write("lr, gamma, pair, source_acc, target_acc, best_score\n")

            hp_best = None
            hp_best_score = None

            for lr, gamma_list in hp_values.items():
                for gamma in gamma_list:
                    hp = HP(lr=lr, gamma=gamma)
                    print("HP:", hp)

                    for source_domain, target_domain in DATASET_PAIRS:
                        _, _, best_score = \
                            train(args, model_name, hp, base_output_dir, hp_file, source_domain, target_domain)
                            
                        if best_score is not None and hp_best_score is None or hp_best_score < best_score:
                            hp_best = hp
                            hp_best_score = best_score

            with open(hp_file, "a") as myfile:
                myfile.write(f"\nbest: {hp_best.lr}, {hp_best.gamma}\n")




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
    parser.add_argument('--results_root', default="./results/")
    parser.add_argument('--model_names', default=["DANN"], nargs='+')
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--hp_tune', default=False, type=bool)
    parser.add_argument('--source', default=None)
    parser.add_argument('--target', default=None) 
    parser.add_argument('--vishook_frequency', default=5, type=int)
       

    args = parser.parse_args()

    print(args)

    for model_name in args.model_names:
        try:
            model_enum = DAModels(model_name)
        except ValueError:
            logging.warning(f"Model {model_name} not found. skipping...")
            continue

        run_experiment_on_model(args, model_enum)

# Tune
# python main.py --max_epochs 10 --patience 3 --trials_count 1 --model_names SYMNET MCD --source amazon --target webcam --hp_tune True --vishook_frequency 10

# Coral
# CUDA_VISIBLE_DEVICES=0 python main.py --max_epochs 60 --patience 10 --trials_count 3 --model_names CORAL --num_workers 1 --batch_size 40
