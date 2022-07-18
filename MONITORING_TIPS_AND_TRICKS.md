# About that document

[FL_PyTorch](https://github.com/burlachenkok/flpytorch) is a software suite based on [PyTorch](https://pytorch.org/) to support efficient simulated Federated Learning experiments.

That document is a cookbook for solving practical problems you may face while using a simulator.

----

### Simulator is running remotely on a remote machine, and I have SSH access, but I can not connect via VNC Viewer. What to do?

It's likely computer administrator blocks the ports. In Linux-based OS, by default, in most cases, people use [iptables](https://linux.die.net/man/8/iptables) to configure acceptance of incoming connections. The first thing you can try to execute from the root is the following command:

```bash
iptables -I INPUT -j ACCEPT
```
If you do not have root access, then you can not execute the previous command. But there is an alternative - using the SSH Port forwarding feature to carry TCP/IP connection artificially through the SSH client.
Please set up SSH port forwarding using the following command:

```bash
ssh -L LOCAL_PORT:DESTINATION:DESTINATION_PORT [USER@]SSH_SERVER
```
or via using the command:
```bash
ssh -L LOCAL_PORT:DESTINATION:DESTINATION_PORT
```

One important thing to mention is that this command may open a local port in a loopback Network Interface Card (NIC). If you want to open a local port available from outside, then use the following command:
```bash
ssh -L 0.0.0.0:LOCAL_PORT:DESTINATION:DESTINATION_PORT
```
or via using the command:
```bash
ssh -L [LOCAL_IP:]LOCAL_PORT:DESTINATION:DESTINATION_PORT
```

----

### I have launched GUI frontend using [starts_with_vnc.sh](https://github.com/burlachenkok/fl_pytorch/blob/main/fl_pytorch/gui/start_with_vnc.sh). How in Linux Machine verify what TCP/IP ports are waiting for incoming connection?
 
For example, you can use the [netstat](https://linux.die.net/man/8/netstat) command combined in the unnamed pipe with [grep](https://linux.die.net/man/1/grep) working in extended regular expression parsing mode:
```bash
netstat -nap | grep -E "tcp(.)*LISTEN"
```
If your process is launched [Python](https://www.python.org/) interpreter, then you can use the following command:
```bash
netstat -nap | grep -E "tcp(.)*LISTEN.*python"
```
----

### Does a version of NVIDIA Driver should be the same on all machines? How to obtain the NVIDIA Driver version?

Unfortunately, the PyTorch project is huge and contains around 5M lines of code. Even with that amount of code, there is significant leverage in the case of using NVIDIA GPU in [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) Libraries.

Suppose you don't have reproducibility of experiments, then maybe reason in different versions of [NVIDIA Driver](https://www.nvidia.com/download/index.aspx).

To obtain the NVIDIA Driver version, use one of the following commands:

```bash
nvidia-smi --query-gpu=driver_version --format=csv
```
or
```bash
nvidia-smi | grep Driver
```

### Can I always use a GPU device for experiments?

For big d, if it is at least 1M parameters, it's better to use GPU typically. But for small models, it's still better to use a CPU. The overhead of transferring data to GPU is far more than just computation in CPU with CPU(host) virtual/physical DRAM memory installed in the machine.


### How to estimate wall clock time for experiments?

In the GUI tool, there are several ways to do that:
* All experiment during execution has in server state statistic *last_round_elapsed_sec* that measures wall clock time for the last round communication round. You can observe that property either in Simulation Tab or add that field to Analysis Tab.
* The second way you can in Analysis Tab can select the "Time per one communication round(minutes) for OX or OY axis."

That time includes selecting clients for participation, client logic for performing optimization logic on the client-side, transferring data to the master, aggregating client information, and performing the model update.

### How to launch FL_PyTorch scripts in a machine safely?
The scripts are launched using a python interpreter, and they need sha-bangs, so in principle, after changing the working directory to the folders with scripts:
```bash
cd  ./fl_pytorch/fl_pytorch/GUI 
```
or
```bash
cd  ./fl_pytorch/fl_pytorch
```

You can launch the simulator in one of the following ways:
```bash
./start.py 
```
or 
```bash
python ./start.py 
```

It will create a Python process in the operating system. In Linux Operation System, the process handles signals from OS. If you instantiated the application on a remote site, then it may be worthwhile to launch:
```bash
session="my"
nohup bash ./start_with_vnc.sh 1>flpytorch_vnc_{session}_stdout.txt 2>flpytorch_vnc_{session}_stderr.txt &
```
In that command [nohup](https://linuxize.com/post/linux-nohup-command/) will guarantee that the [HUP](https://dsa.cs.tsinghua.edu.cn/oj/static/unix_signal.html) signal will be ignored entirely. That signal is sent to the process if the SSH terminal is closed.

### My experiments are too slow. What to do?

It's not easy to answer. The bottleneck can lie in one of the computation/memory transfer aspects. 

The simulator provides enough knobs to tune the computation speed. (And it's primarily due to limits of the applicability of underlying systems under which we rely on)

There are several aspects of tuning the performance speed. First of all, several devices and subsystems are involved in physical computation and memory operations, and it includes:
* Operations in filesystem during reading and writing files.
* CPU DRAM Virtual Memory.
* CPU computations.
* GPU computation underutilization.
* Memory transfer between CPU DRAM memory and GPU DRAM memory.
* Overload of Swap virtual memory system in Operation System(OS).
* Python overhead and Python interpreter implementation limits.

### How to solve problems with CPU computation?
                                                        
During training NN in the current FL_PyTorch version, the preparing data sample points (a,b) data pairs in Optimization Community or (x,y) data pairs in Machine Learning terminology are preprocessed.

It may involve data augmentation that maybe involve random horizontal or vertical flip, random image cropping, convert from NumPy array to PyTorch tensor. That operation is performed on the CPU.

To improve the CPU computation, you can specify "Configuration/System Aspects/Num workers for train set loading" and "Configuration/System Aspects/Num workers for validation set loading."

If you have a multicore CPU, you can select 4,5,6 workers. That leads that while using standard PyTorch DataLoader, the Python process in the context of OS will be forked. 

The time for creating process in Windows OS typically takes roughly around 1 second, and that number in macOS or Linux approximately should have the same order.

It's reasonable to assume that this processing will happen in different CPU cores. To inspect several CPU cores used for computation, you can inspect the output of the following command in Unix based OS'es:

```bash
lscpu
```

Another solution is to use a machine with a more powerful CPU.
