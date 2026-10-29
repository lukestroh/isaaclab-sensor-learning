#!/usr/bin/env python3
import argparse
from time import time
import h5py as h5
import moo.coverage_evaluation as ceval
import moo.instance_parser as iparse
import moo.path_evaluation as peval

import os

import pprint as pp


DATA_DIR = os.path.join(os.path.expanduser("~"), "IsaacLab/source/isaaclab-sensor-learning/data")


def get_nbv_metrics():
    parser = argparse.ArgumentParser(description="Evaluate a NBV run")
    parser.add_argument(
        "--datetime-trial-data-name", type=str, default=None, help="Name of the trial data file."
    )

    args = parser.parse_args()
    datetime_trial_name = args.datetime_trial_data_name
    
    capture_trial_path = os.path.join(DATA_DIR, "capture", f"SL-PDC_{datetime_trial_name}.h5")
    nbv_trial_path = os.path.join(DATA_DIR, "evaluation", f"SL-PDEVAL_{datetime_trial_name}.h5")

    # Get instance label info from capture data trees
    instance_labels = {}
    with h5.File(capture_trial_path, 'r') as file:
        group_trees = file["trees"]
        for env_key, instance_group in group_trees.items():
            if env_key.startswith("env_"):
                instance_labels[env_key] = {}
                for instance_label, instance_id in instance_group.attrs.items():
                    if instance_label.startswith("/World"):
                        instance_labels[env_key][instance_label] = instance_id

    # pp.pprint(instance_labels)
    # Get bin counts from NBV algorithm
    # bin_counts_by_env = {}
    with h5.File(nbv_trial_path, 'r') as file:
        poses = file["pose"][:]
        timestamps = file["timestamp"][:]
        bin_counts = {env_key: data[:] for env_key, data in file["bin_counts"].items()}


    t_final = timestamps[-1]
    peval_metrics = peval.eef_motion_metrics(poses=poses, t_final=t_final)

    ceval_metrics = ceval.coverage_metrics(
        bin_counts=bin_counts,
        instance_map=instance_labels,
        semantic_labels=["trunk", "branch", "spur"],
        parse_instance=iparse.parse_instance,
        min_count=5
    )
    combined_ceval_metrics = ceval_metrics["all"] # get metric across all envs

    nbv_metrics = peval_metrics | combined_ceval_metrics
    return nbv_metrics


def main():
    get_nbv_metrics()
    return


if __name__ == "__main__":
    main()
