#!/usr/bin/env python3

import math, sys, pickle, os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "./../"))
from utils import utils

if __name__ == "__main__":

    files = sys.argv[1:]

    if len(files) == 0:
        print("# Tool for dump information about experiments inside binary files")
        print("# Format: ./dump_configuration.py file1.bin file2.bin file3.bin")

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
            for field, value in H.items():
                if field in ["history", "raw_cmdline", "eval_metrics"]:
                    continue

                print("  ", field, type(value), ": ", str(value))
