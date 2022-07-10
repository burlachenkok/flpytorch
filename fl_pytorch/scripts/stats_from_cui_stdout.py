#!/usr/bin/env python

# Example cmdline: 
#  cat output.txt | run_stats.py

import sys
import numpy as np

data_load_times = []
fw_times   = []
bw_times   = []

# ======================================================================================================================
if __name__ == "__main__":
  for line in sys.stdin:   

    data_load_start = line.find("DataLoad time ")
    if data_load_start >= 0:
      data_load_start += len("DataLoad time ")
      data_load_end = line.find(" ", data_load_start)
      data_load_time = line[(data_load_start):(data_load_end)] 
      data_load_times.append(float(data_load_time))

    fw_start = line.find("F/W time ")
    if fw_start >= 0:
      fw_start += len("F/W time ")
      fw_end = line.find(" ", fw_start)
      fw_time = line[(fw_start):(fw_end)] 
      fw_times.append(float(fw_time))

    bw_start = line.find("B/W time ")
    if bw_start >= 0:
      bw_start += len("B/W time ")
      bw_end = line.find(" ", bw_start)
      bw_time = line[(bw_start):(bw_end)] 
      bw_times.append(float(bw_time))

  # ====================================================================================================================
  print("Total load time {0:0.3f} seconds".format(np.sum(data_load_times)))
  print("  F/W time time {0:0.3f} seconds".format(np.sum(fw_times)))
  print("  B/W time time {0:0.3f} seconds".format(np.sum(bw_times)))
  print("")
  print("Total time to load,F/W,B/W: {0:0.3f} seconds".format(np.sum(data_load_times) + np.sum(fw_times) + np.sum(bw_times)))

