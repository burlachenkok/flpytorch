#!/usr/bin/env python3

import math, sys, pickle, os
import numpy as np
import torch
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), "./../"))
from utils import utils

if __name__ == "__main__":

    files = sys.argv[1:]

    if len(files) == 0:
        print("# Tool for prune away any tensors from history and server state")
        print("# Format: ./prune_tensors.py file1.bin file2.bin file3.bin")
    
    print("Python verions: ", sys.version)
    print("Default pickling protocol: ", pickle.DEFAULT_PROTOCOL)

    for fname in files:
        obj = None
        with open(fname, "rb") as f:
            try:
                obj = pickle.load(f)
            except Exception as ex:
                print("Processing of ", fname, "[FAILED]")
                print("Reason: ", str(ex))
                continue
        
        changes = 0

        for (job_id, H) in obj:
            # Final prune
            HKeys = list(H.keys())

            # Remove any tensor field huger then MBytes threshold
            tensor_prune_mb_threshold = 0.0

            for field in HKeys:

                if torch.is_tensor(H[field]):
                    size_in_mbytes = H[field].numel() * H[field].element_size() / (1024.0 ** 2)
                    if size_in_mbytes >= tensor_prune_mb_threshold:
                        del H[field]
                        changes += 1
                        continue
                    else:
                        # Convert any not removed tensor to CPU
                        H[field] = H[field].cpu()
                        changes += 1
                        continue

                if type(H[field]) == list:
                    sz = len(H[field])
                    i = 0

                    while i < sz:
                        if torch.is_tensor(H[field][i]):
                            size_in_mbytes = H[field][i].numel() * H[field][i].element_size() / (1024.0 ** 2)
                            if size_in_mbytes >= tensor_prune_mb_threshold:
                                del H[field][i]
                                changes += 1
                                sz = len(H[field])
                            else:
                                # Convert any not removed tensor to CPU
                                H[field][i] = H[field][i].cpu()
                                changes += 1
                                i += 1
                        else:
                            i += 1

            for round, history_state  in H['history'].items():
                clients_history = history_state['client_states']
                for client_id, client_state in clients_history.items():
                    client_state = client_state['client_state']
                    client_state_keys = list(client_state.keys())

                    for field in client_state_keys:
                        if torch.is_tensor(client_state[field]):
                            size_in_mbytes = client_state[field].numel() * client_state[field].element_size() / (1024.0 ** 2)
                            if size_in_mbytes >= tensor_prune_mb_threshold:
                                del client_state[field]
                                
                                changes += 1
                                continue
                            else:
                                # Convert any not removed tensor to CPU
                                client_state[field] = client_state[field].cpu()
                                changes += 1
                                continue

                        if type(client_state[field]) == list:
                            sz = len(client_state[field])
                            i = 0                            
                            while i < sz:                          
                                if torch.is_tensor(client_state[field][i]):
                                    size_in_mbytes = client_state[field][i].numel() * client_state[field][i].element_size() / (1024.0 ** 2)
                                    if size_in_mbytes >= tensor_prune_mb_threshold:
                                        del client_state[field][i]
                                        changes += 1
                                        sz = len(client_state[field])
                                    else:                                        
                                        client_state[field][i] = client_state[field][i].cpu()
                                        changes += 1
                                        i += 1
                                else:
                                    i += 1

            try:
                dstfile = fname + ".patched"
                with open(dstfile, "wb") as f:
                    # Use default protocol
                    pickle.dump(copy.deepcopy(obj), f)
                    print("Processing of ", fname, f" with {changes} prunings has been finished. Results in '{dstfile}'. [OK]")

            except Exception as ex:
                print("Processing of", fname, "[FAILED]")
                print("Reason: ", str(ex))
                continue

