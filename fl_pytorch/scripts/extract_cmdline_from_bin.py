#!/usr/bin/env python3

import math, sys, pickle, os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "./../"))
from utils import utils

if __name__ == "__main__":
    python = "python"
    files = sys.argv[1:]
    
    if len(files) == 0:
        print("# Tool for extracting command line from the serialized experiments")
        print("# Format: ./extract_cmdline_from_bin.py file1.bin file2.bin file3.bin")

    for fname in files:
        obj = None
        with open(fname, "rb") as f:
            obj = pickle.load(f)
    
        for (k, v) in obj:
            print()
            print("# Cmdline for ", k)
            cmdline = ""
            for item in v['raw_cmdline']:
               if item.find('--') == 0:
                   cmdline += " "
                   cmdline += item
               else:
                   cmdline += " "
                   cmdline += ('"' + item + '"')

            cmdline = f"{python} run.py" + cmdline
            cmdline = cmdline.strip()
            print(cmdline)

