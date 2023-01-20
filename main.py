import argparse
import logging
import os
from datetime import datetime
from train import train

logging.basicConfig()
logging.getLogger("pytorch-adapt").setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', default=60)
    parser.add_argument('--patience', default=10)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--trials_count', default=3)
    parser.add_argument('--download', default=False)

    args = parser.parse_args()

    root = "."
    data_root = os.path.join(root, "datasets/pytorch-adapt/")


    DATASET_PAIRS = [("amazon", "webcam"), ("amazon", "dslr"),
                    ("webcam", "amazon"), ("webcam", "dslr"),
                    ("dslr", "amazon"), ("dslr", "webcam")]

    model_name = "dann"

    for trial_number in range(4, 6):
        base_output_dir = f"{root}/results/vishook/{model_name}/{trial_number}"
        os.makedirs(base_output_dir, exist_ok=True)

        d = datetime.now()
        results_file = f"{base_output_dir}/{d.strftime('%Y%m%d-%H:%M:%S')}.txt"

        with open(results_file, "w") as myfile:
            myfile.write("pair, source_acc, target_acc, best_epoch, time\n")

        for source_domain, target_domain in DATASET_PAIRS:

            train(args, model_name, data_root, base_output_dir, results_file, source_domain, target_domain)
