#!/usr/bin/env python3

import math, sys, pickle, os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "./../"))
from utils import utils

if __name__ == "__main__":

    files = sys.argv[1:]

    if len(files) == 0:
        print("# Tool for dump information about some tracking quantities")
        print("# Format: ./dump_quality.py file1.bin file2.bin file3.bin")

    for fname in files:
        obj = None
        with open(fname, "rb") as f:
            try:
                obj = pickle.load(f)
            except Exception as ex:
                print("Processing of ", fname, "[FAILED]")
                print("Reason: ", str(ex))
                continue

        for (job_id, H) in obj:
            print("Experiment: ", job_id, " from file: ", fname)

            for field in ["best_metric", "last_round_elapsed_sec"]:
                if field in H:
                    print(f"  {field}: ", H[field])
                else:
                    print(f"  {field}: ", "Not Available")

            if "eval_metrics" in H:
                for field, value in H["eval_metrics"].items():
                    print("  ", field, " round: ", str(value))
            else:
                print("  eval_metrics: ", "Not Available")
