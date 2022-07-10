![Logo](docs/imgs/intro_img.jpg)

# README

# Table of Content

- [About Document](#about-document)
- [Preparation](#preparation)
  * [Optional IDE installation](#optional-ide-installation)
  * [Install Conda environment](#install-conda-environment)
  * [Prepare Conda environment](#prepare-conda-environment)
  * [Install packages via pip and virtualenv](#install-packages-via-pip-and-virtualenv)
  * [Tackling the possible problems in Windows OS](#tackling-the-possible-problems-in-windows-os)
  * [Ability to edit UI files for Qt5](#ability-to-edit-ui-files-for-qt5)
    + [Linux](#linux)
    + [MacOS](#macos)
  * [Clone the project and create your branch](#clone-the-project-and-create-your-branch)
- [Start Work](#start-work)
  * [1. Use GUI](#1-use-gui)
  * [2. You can not install VCN Server but want to use GUI](#2-you-can-not-install-vcn-server-but-want-to-use-gui)
  * [3. Use the console command-line interface](#3-use-the-console-command-line-interface)
- [Obtain Command-Line from GUI](#obtain-command-line-from-gui)
- [Cleaning after Work](#cleaning-after-work)
- [Forum and Q&A](#forum-and-q-a)
- [Reporting an Issue](#reporting-an-issue)
- [Another Resources](#another-resources)
- [About the License](#about-the-license)


# About Document

[FL_PyTorch](https://github.com/burlachenkok/fl_pytorch) is a software suite based on [PyTorch](https://pytorch.org/) to support efficient simulated Federated Learning experiments. This [README.md](README.md) file contains a description of how to prepare and install all needed things to start working with FL_PyTorch.

If you have already installed and prepared all the needed for a start but need assistance in the following steps, we recommend starting with [TUTORIAL.md](TUTORIAL.md). We also provide automatic generated documentation for the code [pdoc3 documentation](docs/generated) located in "docs/generated" folder of the project.

The slides, presentations, posters, and video with the project presentation are available in [presentations](presentations). 

---

# Preparation

## Optional IDE installation
Because [FL_PyTorch](https://github.com/burlachenkok/fl_pytorch) during runtime is presented as a single process, it can be debugged using standard IDE for working with [Python](https://www.python.org/), 
like [PyCharm IDE](https://www.jetbrains.com/pycharm/) or [Visual Studio Code](https://code.visualstudio.com/docs/python/debugging).

So in case of need to debug source code with your changes, we recommend installing some IDE.

## Install Conda environment
If you don't have conda package and environment manager you can install via the following steps for Linux OS:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH="${PATH}:~/miniconda3/bin"
~/miniconda3/bin/conda init bash && source ~/.bashrc && conda config --set auto_activate_base false
```

## Prepare Conda environment
For refresh conda commands, you can look into [official Conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf). 

Commands below will install all needed software as a part of an environment named **fl**, or you can use the name you prefer.

For **Windows OS, Linux OS**, please use the following commands to prepare the environment:

```bash
conda create -n fl python=3.9.1 -y
conda install -n fl pytorch"=1.10.0" torchvision numpy cudatoolkit"=11.1" h5py"=3.6.0" coloredlogs matplotlib psutil pyqt pytest pdoc3 wandb -c pytorch -c nvidia -c conda-forge -y
```

For **Mac OS**, CUDA is not currently available, please install PyTorch without CUDA support:

```bash
conda create -n fl python=3.9.1 -y
conda install -n fl pytorch"=1.10.0" torchvision numpy h5py"=3.6.0" coloredlogs matplotlib psutil pyqt pytest pdoc3 wandb -c pytorch -c nvidia -c conda-forge -y
```


## Install packages via pip and virtualenv

Alternatively, instead of using conda, you can use a standard virtual environment in Python, which sometimes is used instead of Conda.

```bash
python -m pip install virtualenv
python -m venv flpip
source flpip/bin/activate
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
## Tackling the possible problems in Windows OS
In case of any problems with installing PyTorch with CUDA support on a Windows workstation, follow the steps:

1. Install CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
2. Install libCUDNN. You should download and unpack binaries, including all libraries, into the correspondent folders in the local directory with CUDA Toolkit. https://developer.nvidia.com/rdp/cudnn-download
3. Install a specific version of PyTorch with GPU support. For example, in the following way:

```bash
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Ability to edit UI files for Qt5 

In case you want to modify the user interface in the GUI tool via appending extra elements, please make one of the following:

1. Completely install Qt5 SDK, 
2. Install the package with Qt5 tools only, which includes Qt Designer.

### Linux

```bash
sudo apt-get install qttools5-dev-tools
```
Apt is a standard package management tool in the Ubuntu OS. If you have never worked with it but want to learn more, this is the best place with various how to:
[apt-get: How To](https://help.ubuntu.com/community/AptGet/Howto).

### MacOS

```bash
brew install qt5
brew install --cask qt-creator
```

## Clone the project and create your branch

Install git client for your OS if it has not been installed, and clone the project:

```bash
# Clone project
git clone https://github.com/burlachenkok/fl_pytorch
cd fl_pytorch

# Create a personal branch
git branch my_work
git checkout my_work
git push --set-upstream origin my_work
```

There is nothing to compile or pre-build. The project is entirely has been written in [Python](https://www.python.org/) programming language and does not contain any [C++](https://en.cppreference.com/) or [Cython](https://cython.readthedocs.io/en/latest/index.html) parts in it explicitly.

# Start Work

You have three options to start working. 
## 1. Use GUI

Suppose you interconnect with a physical/remote computing device with a windows management system in such a way you can instantiate an application with a GUI interface.
In that case, you can launch our simulation environment with Graphical User Interface(GUI) interface to experiment with optimization algorithms:

```bash
cd ./fl_pytorch/GUI 
python start.py
```
After that, you will observe the GUI tool. GUI tool allows launching several experiments simultaneously. The entry point to use the GUI tool launch script is located here [fl_pytorch/GUI/start.py](fl_pytorch/gui/start.py)

## 2. You can not install VCN Server but want to use GUI

Another scenario may happen when you have access to the machine in a cluster with dedicated installed GPU/CPU compute resources. 
If you don't have root privileges to install the Windows Management system native for OS or provided as VNC systems like [*tigervncserver*](https://tigervnc.org/), we provide the ability to launch the GUI tool with the following commands:

```bash
cd ./fl_pytorch/fl_pytorch/GUI 
./start_with_vnc.sh
```
To change the listening port, please directly change *listet_port* parameter inside this bootstrap bash script. 
To connect remotely for an application instance, please use your host machine and VNC Viewer, e.g. [RealVNC VNC Viewer](https://www.realvnc.com/en/).
The booststrap script to launch VNC server is locating here: [fl_pytorch/gui/start_with_vnc.sh](fl_pytorch/gui/start_with_vnc.sh).

## 3. Use the console command-line interface

The last option is to use the console interface for the experiment. Any research project has many options to select, and our project is not an exception to this rule, but we provide some assistance on that aspect. 
To observe options for the CLI program, please launch:

```bash
cd ./fl_pytorch/ 
python ./run.py --help
```
The entry point and the script to use during work with CUI are located here [fl_pytorch/run.py](fl_pytorch/run.py).

# Obtain Command Line from GUI

* The first way is to specify command-line parameters explicitly via using ```python ./run.py ...```
* Second way is to launch GUI tool ```python start.py```, select Extra->Low Window(F2), and press one of the command line icons. It will produce a command line for loaded and configured experiments in GUI.

# Cleaning after Work

Remove Conda environment in case you are using Conda: 
```bash
conda remove --name fl --all
```
And please remove all not need files in the filesystem, including logs, checkpoints, and saved experiments in binary format.

# Forum and Q&A

You are welcome to join the Slack workspace of project and ask questions.

https://fl-pytorch-workspace.slack.com/

# Reporting an Issue

For bug or feature request, please fill the ticket in the github repository.

# Another Resources

1. [https://arxiv.org/abs/2202.03099](https://arxiv.org/abs/2202.03099) 
2. [https://dl.acm.org/doi/abs/10.1145/3488659.3493775](https://dl.acm.org/doi/abs/10.1145/3488659.3493775)
3. [Video. FL_PyTorch at Rising Stars in AI Symposium 2022 at KAUST](https://webcast.kaust.edu.sa/mediasite/Showcase/kaust/Presentation/600c852bedf94c8298f92d8c1703f8521d)
4. [FL_PyTorch in deepai.org](https://deepai.org/publication/fl-pytorch-optimization-research-simulator-for-federated-learning)
5. [FL_PyTorch in researchgate.net](https://www.researchgate.net/publication/358422816_FL_PyTorch_optimization_research_simulator_for_federated_learning)

# About the License

The project is distributed under [Apache 2.0 license](LICENSE).
