#!/usr/bin/env python3

# https://wiki.python.org/moin/PyQt
# https://www.riverbankcomputing.com/static/Docs/PyQt5/

# Example of path to designer in Windows OS: Python38/Lib/site-packages/qt5_applications/Qt/bin/designer.exe

import sys, platform, time, shutil, pickle, threading, math, socket
import datetime
import os
import psutil
import gc

import random
import copy
import numpy as np
import torch
import logging

# Qt namespace is a full list of attributes that you can use to customize and control Qt widgets
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox, QDialog, QFileDialog, QFontDialog
from PyQt5 import QtCore, QtGui, QtWidgets

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

sys.path.append(os.path.join(os.path.dirname(__file__), "./generated"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./"))

from generated import MainView, AboutBox, ConfigWidget, LogWindow,\
                      SimulationWidget, AnalysisWidget, MultiMachineSelector

from utils import gpu_utils, comm_socket, git_utils, algorithms
import utils
import run
import data_preprocess

# Import PyTorch root package import torch
import torch

import matplotlib
from torch.utils.collect_env import get_pretty_env_info

# Matplotlib functionality
import matplotlib

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# Import the toolbar widget for matplotlib
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Configure that plots from Matplotlib displayed in PyQt5 are actually rendered by the Agg backend.
matplotlib.use("Qt5Agg")


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.fig = fig
        super().__init__(fig)

        # Instantiate toolbar. First arg - is canvas for rendering, and second is parent of toolbar.
        self.toolbar = NavigationToolbar(self, parent)


app = None                            # Application object
mainView = None                       # Main window
logView = None                        # Log window
configView = None                     # Configuration window
simulationView = None                 # Widget for simulation
analysisView = None                   # Widget for analysis


simulationThreadPool = utils.thread_pool.ThreadPool()             # Simulation main threads


sim_stats_res_lock = threading.Lock()    # Locker for next 4 data objects
earlyStopSimulations = set()             # Set of early stop simulations job_id
simulations_stats = {}                   # Simulations key - job_id, value collected statistics
simulations_result = {}                  # Simulation result, key - job_id, value server-state (stop to change)
simulations_by_rows = []                 # Simulations, index - row in table, value - job_id

gui_rnd = random.Random()                # Random generator in UI

machineDescriptions = []                 # Description of external used compute resources

# ======================================================================================================================
compressorTypes = ["ident", "randk",
                   "bernulli", "natural",
                   "qsgd", "nat.dithering", "std.dithering", "topk", 
                   "rank_k",
                   "terngrad"]

compressorFormats = ["", "randk:<percentage-of-D [0,100]>% | randk:k[1,D]",
                     "bernulli:p[0,1]", "",
                     "qsgd:levels", "nat.dithering:levels:norm",  "std.dithering:levels:norm",
                     "topk:<percentage-of-D [0,100]>% | topk:<K>[1,D]",
                     "rank_k:<percentage-of-D [0,100]>% | rank_k:<K>[1,D]",
                     ""]
# ======================================================================================================================
requestUpdateResultTable = False


def showArtificialDatasetParameters(show=True):
    configView.lblL.setVisible(show)
    configView.edtL.setVisible(show)
    configView.lblMu.setVisible(show)
    configView.edtMu.setVisible(show)

    if show:
        configView.edtNumberOfClients.setReadOnly(False)
        configView.edtNumberOfClients.setStyleSheet("background-color: rgb(255, 255, 255);")
        configView.lblGlobalModel.setVisible(False)
        configView.cbxGlobalModel.setVisible(False)
        configView.lblLossFunctionForGlobalModel.setVisible(False)
        configView.cbxLossFunctionForGlobalModel.setVisible(False)
        configView.lblSamplesPerClient.setVisible(True)
        configView.edtSamplesPerClient.setVisible(True)
        configView.lblHomogeneousDS.setVisible(True)
        configView.cbxHomogeneousDS.setVisible(True)
        configView.lblValidationOptMetric.setVisible(False)
        configView.cbxValidationOptMetric.setVisible(False)
        configView.lblVarsInOpt.setVisible(True)
        configView.edtVarsInOpt.setVisible(True)

    else:
        configView.edtNumberOfClients.setReadOnly(True)
        configView.edtNumberOfClients.setStyleSheet("background-color: rgb(233, 185, 110);")
        configView.lblGlobalModel.setVisible(True)
        configView.cbxGlobalModel.setVisible(True)
        configView.lblLossFunctionForGlobalModel.setVisible(True)
        configView.cbxLossFunctionForGlobalModel.setVisible(True)
        configView.lblSamplesPerClient.setVisible(False)
        configView.edtSamplesPerClient.setVisible(False)
        configView.lblHomogeneousDS.setVisible(False)
        configView.cbxHomogeneousDS.setVisible(False)
        configView.lblValidationOptMetric.setVisible(True)
        configView.cbxValidationOptMetric.setVisible(True)
        configView.lblVarsInOpt.setVisible(False)
        configView.edtVarsInOpt.setVisible(False)


def totalSimulations():
    res = 0
    for th in simulationThreadPool.threads:
        res += len(th.cmds)
    return res


def onMoveToNextTabInMain():
    """Move to the next tab in a cyclic way in a tabs inside a main window"""
    mainView.tabMainView.setCurrentIndex((mainView.tabMainView.currentIndex() + 1) % mainView.tabMainView.count())


def onMoveToPrevTabInMain():
    """Move to the prev tab in a cyclic way in a tabs inside a main window"""
    prev = mainView.tabMainView.currentIndex() - 1
    while prev < 0:
        prev += mainView.tabMainView.count()
    mainView.tabMainView.setCurrentIndex(prev)


def onShowAboutDialog():
    """Show about dialog"""
    about_box = QDialog()
    about_box_ui = AboutBox.Ui_AboutBox()
    about_box_ui.setupUi(about_box)
    about_box.ui = about_box_ui

    about_box_ui.btnCloseAbout.clicked.connect(lambda: about_box.close())

    # Update information about the program
    programInfo = "FL_PyTorch Optimization Research Tool, 2022.\n"
    programInfo += "\n"
    programInfo += f"Branch name: {git_utils.branch()}\n"
    programInfo += f"Revision: {git_utils.revision()}\n"
    programInfo += f"Data of last revision submission: {git_utils.dateAndTimeOfLastRevision()}\n"
    programInfo += "\n"
    programInfo += f"Current working directory: {os.getcwd()}\n"
    programInfo += f"Executed program: {__file__}\n"
    programInfo += "\n"
    programInfo += f"Python: {sys.executable}\n"
    programInfo += f"Python version: {sys.version}\n"
    programInfo += f"Platform name: {sys.platform}\n"
    programInfo += "\n"
    programInfo += f"PyTorch version: {torch.__version__}\n"
    programInfo += f"Matplotlib version: {matplotlib.__version__}\n"
    programInfo += f"Qt version: {QtCore.qVersion()}\n"
    programInfo += f"NumPy version: {np.__version__}\n"
    programInfo += "\n"
    programInfo += "Repository: https://github.com/burlachenkok/flpytorch\n"
    programInfo += "Paper: https://dl.acm.org/doi/abs/10.1145/3488659.3493775"

    about_box_ui.edtText.setText(programInfo)
    about_box.show()
    about_box.exec_()


def uiLogInfo(text):
    prevText = logView.txtMain.toPlainText()
    logView.txtMain.setPlainText(prevText + text + "\n")


def onMemoryInfo():
    process = psutil.Process(os.getpid())
    uiLogInfo(f'Allocated resident host (CPU) memory {process.memory_info().rss / (1024 ** 2):.2f} MBytes')
    uiLogInfo(f'Reserved virtual host (CPU) memory {process.memory_info().vms / (1024 ** 2):.2f} MBytes')
    mem = psutil.virtual_memory()
    uiLogInfo(f'Available physical RAM memory in a system {mem.total / (1024**3):.2f} GBytes')

    gpus_properties = gpu_utils.get_available_gpus()
    uiLogInfo(f"Number of installed GPUs in the system: {len(gpus_properties)}")
    for i in range(len(gpus_properties)):
        device = "cuda:" + str(i)
        memory_gpu = torch.cuda.memory_stats(device)['reserved_bytes.all.current']
        uiLogInfo("  {0}. Used {1:.2f} MBytes. Available {2:.2f} MBytes GDDR".format(gpus_properties[i].name,
                                                                                     memory_gpu/1024.0/1024.0,
                                                                                     gpus_properties[i].total_memory /
                                                                                     (1024. ** 2)
                                                                                     )
                  )


def onExitLogWindow():
    """Close log window"""
    logView.close()


def onExitMachineSelectorWindow():
    """Close machine selector window"""
    machinesView.close()


def onCmdLineGenerationForCurrentExperiment(withNewLines):
    """Generate command line for current experiment"""
    python = "python"

    if True:
        cmds = currentCommandLineFromGUI()
        job_id = "current"

        cmdline = ""
        cmdline_new_line = " \\"  # For Bash

        for item in cmds:
            item = item.strip()

            if item.find("--") == 0:
                # Move to new line
                if withNewLines:
                    cmdline += cmdline_new_line + "\n" + item
                else:
                    cmdline += " " + item

            elif len(item) == 0:
                # Empty argument - escaping with ""
                cmdline += " "
                cmdline += '""'
            else:
                # Extra escaping for possible arguments
                cmdline += " "
                cmdline += ('"' + item + '"')

        # Add output filename for cmdline
        if withNewLines:
            cmdline += (cmdline_new_line + "\n" + f'--out "{job_id}.bin"')
        else:
            cmdline += (" " + f'--out "{job_id}.bin"')

        cmdline = f"{python} run.py" + cmdline
        cmdline = cmdline.strip()

        uiLogInfo(f"# ===========================================================")
        uiLogInfo(f"{datetime.datetime.now()}")
        uiLogInfo(f"# ===========================================================")
        uiLogInfo(f"# Command line for currently selected configuration in GUI")
        uiLogInfo(cmdline)
        uiLogInfo(f"# ===========================================================")
        uiLogInfo("")


def onCmdLineGenerationForFinishedExperiments(withNewLines):
    """Generate command line for finished experiments and current experiment"""
    python = "python"

    for job_id in simulations_stats.keys():
        sim_stats_res_lock.acquire()
        cmds = simulations_stats[job_id]['H']['raw_cmdline']
        sim_stats_res_lock.release()

        cmdline = ""
        cmdline_new_line = " \\"  # For Bash

        for item in cmds:
            item = item.strip()

            if item.find("--") == 0:
                # Move to new line
                if withNewLines:
                    cmdline += cmdline_new_line + "\n" + item
                else:
                    cmdline += " " + item

            elif len(item) == 0:
                # Empty argument - escaping with ""
                cmdline += " "
                cmdline += '""'
            else:
                # Extra escaping for possible arguments
                cmdline += " "
                cmdline += ('"' + item + '"')

        # Add output filename for cmdline
        if withNewLines:
            cmdline += (cmdline_new_line + "\n" + f'--out "{job_id}.bin"')
        else:
            cmdline += (" " + f'--out "{job_id}.bin"')

        cmdline = f"{python} run.py" + cmdline
        cmdline = cmdline.strip()
        uiLogInfo(f"# Command line for experiment with job_id={job_id}")
        uiLogInfo(cmdline)
        uiLogInfo("")


def onExperimentInfoGeneration():
    """Generate description information about current numerical experiment"""
    algorithm = configView.cbxOptAlgo.currentText().lower()
    docString = algorithms.getImplClassForAlgo(algorithm).__doc__
    docString = docString.replace("\r\n", "").replace("\n", "")
    uiLogInfo("Optimization Algorithm:")
    uiLogInfo(docString)


def onCleanupMemory():
    process = psutil.Process(os.getpid())
    memory_cpu_start = (process.memory_info().rss + process.memory_info().vms)  # in bytes

    gpus_properties = gpu_utils.get_available_gpus()
    memory_gpu_start = []
    memory_gpu_end = []

    for i in range(len(gpus_properties)):
        device = "cuda:" + str(i)
        # get the current active allocated memory in bytes
        memory_gpu_start.append(torch.cuda.memory_stats(device)['reserved_bytes.all.current'])

    # do the garbage collection
    uiLogInfo('Release unoccupied cache memory from PyTorch...')
    torch.cuda.empty_cache()

    uiLogInfo('Running the garbage collector...')
    gc.collect()  # gc is the garbage collection module

    memory_cpu_end = (process.memory_info().rss + process.memory_info().vms)  # in bytes
    memory_freed = abs(memory_cpu_start - memory_cpu_end)
    uiLogInfo(f' Done. {memory_freed / 1024 ** 2:.2f} MB was removed from Virtual and Resident memory of interpreter. Current used amount of memory is {memory_cpu_end/ 1024 ** 2:.2f} MBytes')

    for i in range(len(gpus_properties)):
        device = "cuda:" + str(i)
        # get the current active allocated memory in bytes
        memory_gpu_end.append(torch.cuda.memory_stats(device)['reserved_bytes.all.current'])
        memory_freed = memory_gpu_start[i] - memory_gpu_end[i]
        uiLogInfo(f' Done. {memory_freed / 1024 ** 2:.2f} MB was removed from {device}. Current used amount of memory is {memory_gpu_end[-1]/ 1024 ** 2:.2f} MBytes')


def onLogSystemInformation():
    """Append some default information into log window"""
    uiLogInfo("Information about Python")
    uiLogInfo(f"  Path to python: {sys.executable}")
    uiLogInfo(f"  Python version: {sys.version}")
    uiLogInfo(f"  Platform name: {sys.platform}")
    uiLogInfo(f"  Current working directory: {os.getcwd()}")
    uiLogInfo("")
    uiLogInfo("Information about System")
    (system, node, release, version, machine, processor) = platform.uname()
    uiLogInfo(f"  System/OS name: {system}/{release}/{version}")
    uiLogInfo(f"  Machine name: {machine}")
    uiLogInfo(f"  Host name: {socket.gethostname()}")
    uiLogInfo(f"  IP address of one Network Interface: {socket.gethostbyname(socket.gethostname())}")

    uiLogInfo("")
    uiLogInfo("Information about installed compute devices")
    uiLogInfo(f"  CPU name: {processor}")
    gpus_properties = gpu_utils.get_available_gpus()
    uiLogInfo(f"  Number of installed GPUs in the system: {len(gpus_properties)}")
    for i in range(len(gpus_properties)):
        uiLogInfo("  {0} {1:g} GBytes of GDDR".format(gpus_properties[i].name, gpus_properties[i].total_memory / (1024.0 ** 3)))
    uiLogInfo("")
    uiLogInfo("Information about installed software")
    uiLogInfo(f"  PyTorch version: {torch.__version__}")
    uiLogInfo(f"  Matplotlib version: {matplotlib.__version__}")
    uiLogInfo(f"  Qt version: {QtCore.qVersion()}")
    uiLogInfo(f"  NumPy version: {np.__version__}")


def isSimulationNeedEarlyStop(H):
    """Predictor which provide information do we need early stopping"""
    global earlyStopSimulations
    global sim_stats_res_lock

    result = False

    sim_stats_res_lock.acquire()
    if H['run_id'] in earlyStopSimulations:
        result = True                               # Report that this simulation need early stop
        earlyStopSimulations.discard(H['run_id'])   # Remove job_id for simulation
    sim_stats_res_lock.release()

    return result

def simulationProgressSteps(progress, H):
    job_id = H["run_id"]

#    dataload_duration = []
#    inference_duration = []
#    backprop_duration = []
#    full_gradient_oracles = []
#    samples_gradient_oracles = []
#    send_scalars_to_master = []

    # =================================================================================================================
    for round, round_info in H['history'].items():
        for client_id, client_summary in round_info['client_states'].items():
            client_stats = client_summary['client_state']['stats']

#            dataload_duration.append(client_stats['dataload_duration'])
#            inference_duration.append(client_stats['inference_duration'])
#            backprop_duration.append(client_stats['backprop_duration'])
#            full_gradient_oracles.append(client_stats['full_gradient_oracles'])
#            samples_gradient_oracles.append(client_stats['samples_gradient_oracles'])
#            send_scalars_to_master.append(client_stats['send_scalars_to_master'])
    # ==================================================================================================================

    sim_stats_res_lock.acquire()
    simulations_stats[job_id]["progress"] = int(100 * progress)
    rounds = H['history'].keys()

    simulations_stats[job_id]["completed_rounds"] = len(rounds)
    simulations_stats[job_id]["best_metric"] = H["best_metric"]

    if 'th_stepsize_noncvx' in H:
        simulations_stats[job_id]["th_stepsize_noncvx"] = H['th_stepsize_noncvx']

    if 'th_stepsize_cvx' in H:
        simulations_stats[job_id]["th_stepsize_cvx"] = H['th_stepsize_cvx']

    if len(rounds) > 0:
        current_round = max(rounds)
        simulations_stats[job_id]["grad_sgd_server_l2"] = H['history'][current_round]['grad_sgd_server_l2']
        simulations_stats[job_id]["approximate_f_avg_value"] = H['history'][current_round]['approximate_f_avg_value']

    # if len(dataload_duration) > 0:
    #   simulations_stats[job_id]["dataload_duration_avg"] = sum(dataload_duration)/len(dataload_duration)
    # if len(inference_duration) > 0:
    #   simulations_stats[job_id]["inference_duration_avg"] = sum(inference_duration)/len(inference_duration)
    # if len(backprop_duration) > 0:
    #   simulations_stats[job_id]["backprop_duration_avg"] = sum(backprop_duration)/len(backprop_duration)
    # if len(full_gradient_oracles) > 0:
    #   simulations_stats[job_id]["full_gradient_oracles_avg"] = sum(full_gradient_oracles)/len(full_gradient_oracles)
    # if len(samples_gradient_oracles) > 0:
    #   simulations_stats[job_id]["samples_gradient_oracles_avg"] = sum(samples_gradient_oracles)/len(samples_gradient_oracles)
    # if len(send_scalars_to_master) > 0:
    #   simulations_stats[job_id]["send_scalars_to_master"] =  sum(send_scalars_to_master)/len(send_scalars_to_master)

    simulations_stats[job_id]["H"] = H

    if "server_state_update_time" in H:
        simulations_stats[job_id]["server_state_update_time"] = H["server_state_update_time"]
    if "last_round_elapsed_sec" in H:
        simulations_stats[job_id]["last_round_elapsed_sec"] = H["last_round_elapsed_sec"]

    sim_stats_res_lock.release()
    # ==================================================================================================================
    # Make request for update table for analysis
    global requestUpdateResultTable
    requestUpdateResultTable = True
    # ==================================================================================================================


def simulationFinishInternal(H, lock):
    job_id = H["run_id"]

    if lock:
        sim_stats_res_lock.acquire()

    earlyStopSimulations.discard(H['run_id'])

    simulations_result[job_id] = H

    for item in vars(H["args"]):
        simulations_stats[job_id][item] = getattr(H["args"], item)

    if 'th_step_size_cvx' in H:
        simulations_stats[job_id]["th_step_size_cvx"] = H['th_step_size_cvx']

    if 'th_step_size_nocvx' in H:
        simulations_stats[job_id]["th_step_size_nocvx"] = H['th_step_size_nocvx']

    simulations_stats[job_id]["best_metric"] = H["best_metric"]
    simulations_stats[job_id]["D"] = H["D"]
    simulations_stats[job_id]["D_include_frozen"] = H["D_include_frozen"] # dimension of the problem including frozen variables(not trainable)
    simulations_stats[job_id]["progress"] = 100
    simulations_stats[job_id]["completed_rounds"] = len(H['history'])
    simulations_stats[job_id]["finished"] = True

    # ==================================================================================================================
    simulations_stats[job_id]["client_compressor"] = H["client_compressor"]

    if "xfinal" in H:
        simulations_stats[job_id]["xfinal"] = H["xfinal"]

    if "used_x_solution" in H:
        simulations_stats[job_id]["used_x_solution"] = H["used_x_solution"]

    if "server_state_update_time" in H:
        simulations_stats[job_id]["server_state_update_time"] = H["server_state_update_time"]

    if "last_round_elapsed_sec" in H:
        simulations_stats[job_id]["last_round_elapsed_sec"] = H["last_round_elapsed_sec"]

    simulations_stats[job_id]["args"] = H["args"]
    if "group-name" in H:
        simulations_stats[job_id]["group-name"] = H["group-name"]
    # ==================================================================================================================
    rounds = H['history'].keys()
    if len(rounds) > 0:
        current_round = max(rounds)
        simulations_stats[job_id]["grad_sgd_server_l2"] = H['history'][current_round]['grad_sgd_server_l2']
        #simulations_stats[job_id]["xi_after"] = [H['history'][k]['xi_after'] for k in H['history'].keys()]
        simulations_stats[job_id]["approximate_f_avg_value"] = H['history'][current_round]['approximate_f_avg_value']
    simulations_stats[job_id]["H"] = H

    if lock:
        sim_stats_res_lock.release()

    # Force updating table with results
    global requestUpdateResultTable
    requestUpdateResultTable = True


def simulationFinish(H):
    return simulationFinishInternal(H, lock = True)


def simulationStart(H):
    job_id = H["run_id"]
    stats = {"progress": 0, "finished": False}
    # copy all args for launching
    for item in vars(H["args"]):
        stats[item] = getattr(H["args"], item)

    stats["D"] = H["D"]                                # dimension of the problem
    stats["D_include_frozen"] = H["D_include_frozen"]  # dimension of the problem including frozen variables(not trainable)
    stats["completed_rounds"] = 0                      # number of completed rounds
    stats["finished"] = False                          # flag about the fact that job is finished

    stats["client_compressor"] = H["client_compressor"]

    if "start_time" in H:
        stats["start_time"] = H["start_time"]

    if "server_state_update_time" in H:
        stats["server_state_update_time"] = H["server_state_update_time"]

    if "last_round_elapsed_sec" in H:
        simulations_stats[job_id]["last_round_elapsed_sec"] = H["last_round_elapsed_sec"]

    stats["args"] = H["args"]
    stats["progress"] = int(0)
    stats["H"] = H

    if "group-name" in H:
        stats["group-name"] = H["group-name"]

    if "x0" in H:
        stats["x0_l2_norm"] = torch.linalg.norm(H["x0"])
        stats["x0"] = H["x0"]

    if "xfinal" in H:
        stats["xfinal"] = H["xfinal"]

    # stats["dataload_duration_avg"]     = 0
    # stats["inference_duration_avg"]    = 0
    # stats["backprop_duration_avg"]     = 0
    # stats["full_gradient_oracles_avg"] = 0
    # stats["samples_gradient_oracles_avg"] = 0
    # stats["send_scalars_to_master"]       = 0

    sim_stats_res_lock.acquire()
    simulations_stats[job_id] = stats
    simulations_by_rows.append(job_id)
    sim_stats_res_lock.release()

    # Make request for update table for analysis
    global requestUpdateResultTable
    requestUpdateResultTable = True


# ======================================================================================================================
class ProgressDelegate(QtWidgets.QStyledItemDelegate):
    """Progress bar delegate to experiment status"""
    def paint(self, painter, option, index):
        progress, is_terminated_early = index.data(QtCore.Qt.UserRole + 1000)
        opt = QtWidgets.QStyleOptionProgressBar()
        opt.rect = option.rect
        opt.minimum = 0
        opt.maximum = 100
        opt.progress = progress
        opt.text = "{}%".format(progress)
        opt.textVisible = True

        if is_terminated_early:
            opt.palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(157, 13, 20))

        QtWidgets.QApplication.style().drawControl(QtWidgets.QStyle.CE_ProgressBar, opt, painter)


# ======================================================================================================================
class StatusDelegate(QtWidgets.QStyledItemDelegate):
    """Status delegate which is dedicated to report about remote machine status online/offline"""

    def paint(self, painter, option, index):
        online = index.data(QtCore.Qt.UserRole + 2000)
        brush = QtGui.QBrush()
        brush.setStyle(Qt.SolidPattern)
        rect = QtCore.QRect(option.rect)

        if online:
            brush.setColor(QtGui.QColor("green"))
            painter.fillRect(rect, brush)
            painter.drawText(option.rect, Qt.AlignLeft, "online")
        else:
            brush.setColor(QtGui.QColor("grey"))
            painter.fillRect(rect, brush)
            painter.drawText(option.rect, Qt.AlignLeft, "offline")


# ======================================================================================================================
def onUpdateStatusForExternalResources():
    """Update connection status for a remote resources"""

    # Retrieve information from the network
    for i in range(len(machineDescriptions)):
        descr = machineDescriptions[i]

        try:
            s = comm_socket.CommSocket()
            # Use 5 socket timeout to setup a connection with a remote side
            s.sock.settimeout(5)
            s.sock.connect((descr["ip"], descr["port"]))
            descr["online"] = True
            s.rawSendString("list_of_gpus")
            gpu_count = int(s.rawRecvString())
            descr["devices"] = ["cpu:-1"]
            for i in range(gpu_count):
                descr["devices"].append(f"gpu:{i}")

        except socket.error as exc:
            # print(exc)
            descr["devices"] = []
            descr["online"] = False
    # ==================================================================================================================
    # Fill the table
    for r in range(machinesView.tblMachines.rowCount()):
        descr = machineDescriptions[r]

        it_name = machinesView.tblMachines.item(r, 0)
        it_ip = machinesView.tblMachines.item(r, 1)
        it_port = machinesView.tblMachines.item(r, 2)
        it_gpus = machinesView.tblMachines.item(r, 3)
        it_use_cpu = machinesView.tblMachines.item(r, 4)
        it_use_gpu1 = machinesView.tblMachines.item(r, 5)
        it_use_gpu2 = machinesView.tblMachines.item(r, 6)
        it_use_gpu3 = machinesView.tblMachines.item(r, 7)
        it_use_gpu4 = machinesView.tblMachines.item(r, 8)
        it_online = machinesView.tblMachines.item(r, 9)

        it_gpus.setText(str(len([d for d in descr["devices"] if d.find("cpu") == -1])))
        it_online.setData(QtCore.Qt.UserRole + 2000, descr["online"])

        if "cpu:-1" in descr["devices"]:
            it_use_cpu.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
        else:
            it_use_cpu.setFlags(QtCore.Qt.NoItemFlags)

        if "gpu:0" in descr["devices"]:
            it_use_gpu1.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
        else:
            it_use_gpu1.setFlags(QtCore.Qt.NoItemFlags)

        if "gpu:1" in descr["devices"]:
            it_use_gpu2.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
        else:
            it_use_gpu2.setFlags(QtCore.Qt.NoItemFlags)

        if "gpu:2" in descr["devices"]:
            it_use_gpu3.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
        else:
            it_use_gpu3.setFlags(QtCore.Qt.NoItemFlags)

        if "gpu:3" in descr["devices"]:
            it_use_gpu4.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
        else:
            it_use_gpu4.setFlags(QtCore.Qt.NoItemFlags)
    # ==================================================================================================================
    # Force update view for machines
    machinesView.tblMachines.update()
    # ==================================================================================================================


def onTimerEvent():
    w = simulationView.tblExperiments
    sim_stats_res_lock.acquire()
    while w.rowCount() < len(simulations_by_rows):
        it_id = QtWidgets.QTableWidgetItem("")
        it_device = QtWidgets.QTableWidgetItem("")
        it_progress = QtWidgets.QTableWidgetItem()

        w.insertRow(w.rowCount())
        r = w.rowCount() - 1

        it_progress.setFlags(it_progress.flags() ^ QtCore.Qt.ItemIsEditable)
        it_device.setFlags(it_device.flags() ^ QtCore.Qt.ItemIsEditable)

        for c, item in enumerate((it_id, it_device, it_progress)):
            w.setItem(r, c, item)

    for i in range(len(simulations_by_rows)):
        job_id = simulations_by_rows[i]
        progress = simulations_stats[job_id]["progress"]
        device_id = simulations_stats[job_id]["gpu"]
        # For comfortable of GUI user represent single device as single string
        if type(device_id) is list and len(device_id) == 1:
            device_id = device_id[0]

        w.item(i, 0).setText(job_id)
        w.item(i, 1).setText(str(device_id))

        is_terminated_early = False
        if simulations_stats[job_id]["finished"] and \
                simulations_stats[job_id]["completed_rounds"] < simulations_stats[job_id]["rounds"]:
            is_terminated_early = True

        w.item(i, 2).setData(QtCore.Qt.UserRole + 1000, (progress, is_terminated_early) )

    if totalSimulations() == 0:
        # Disable remove experiment button
        simulationView.btnRemoveExperiments.setEnabled(True)
        simulationView.btnClean.setEnabled(True)
    else:
        # Enable remove experiment button
        simulationView.btnRemoveExperiments.setEnabled(False)
        simulationView.btnClean.setEnabled(False)

    sim_stats_res_lock.release()

    while w.rowCount() > len(simulations_by_rows):
        w.removeRow(w.rowCount() - 1)

    updateMemoryUsageStatus()

    global requestUpdateResultTable

    if requestUpdateResultTable:
        onUpdateTableForAnalysis()
        requestUpdateResultTable = False
    # ==================================================================================================================
    # Update titles of tabs in analysis widget
    tab_index = analysisView.tabWidget.currentIndex()

    if analysisView.cbxSyncTitleWithTabName.checkState() == Qt.Checked:
        setup_text = analysisView.customPlots[tab_index].axes.title.get_text()
    else:
        setup_text = f"Plot-{tab_index+1}"

    if setup_text != analysisView.tabWidget.tabText(tab_index):
        analysisView.tabWidget.setTabText(tab_index, setup_text)
    # ==================================================================================================================

def cbxClientSamplingTypeSelection(index):
    algo = configView.cbxClientSamlingType.currentText().lower()
    if algo == "uniform sampling":
        configView.lblPoissonSampling.setVisible(False)
        configView.edtPoissonSampling.setVisible(False)
        configView.lblNumClientsPerRound.setVisible(True)
        configView.edtNumClientsPerRound.setVisible(True)
    elif algo == "poisson sampling" or algo == "poisson sampling with no empty sample":
        configView.lblPoissonSampling.setVisible(True)
        configView.edtPoissonSampling.setVisible(True)
        configView.lblNumClientsPerRound.setVisible(False)
        configView.edtNumClientsPerRound.setVisible(False)


def cbxOptAlgorithmSelection(index):
    algo = configView.cbxOptAlgo.currentText().lower()
    #if algo == "fedprox":
    #    configView.edtExperimentalExtraOpts.setText("mu_prox:0.1")

def cleanupFromComments(input, startComment = "(", endComment = ")"):
    '''Remove comments in form (my comment), and remove beginning and ending whitespaces '''
    while True:
        s = input.find("(")
        e = input.find(")")
        if s == -1 or e == -1:
            break
        else:
            input = input[:max(0,s)] + input[e+1:]
    return input.strip()

def currentCommandLineFromGUI():
    """Generate comman line arguments"""

    # Automatically generate seeds
    if configView.cbxGenerateInitSeedAuto.checkState() == Qt.Checked:
        configView.edtRandomInitSeed.setText(str(gui_rnd.randint(1, 10**9)))

    if configView.cbxGenerateRunSeedAuto.checkState() == Qt.Checked:
        configView.edtRandomRunSeed.setText(str(gui_rnd.randint(1, 10**9)))

    # Arguments for command line
    cmdline = []

    cmdline.append("--rounds")
    cmdline.append(configView.edtComRounds.text())

    sampling_index_type = configView.cbxClientSamlingType.currentIndex()
    if sampling_index_type == 0:
        cmdline.append("--client-sampling-type")
        cmdline.append("uniform")
        cmdline.append("--num-clients-per-round")
        cmdline.append(configView.edtNumClientsPerRound.text())
    elif sampling_index_type == 1:
        cmdline.append("--client-sampling-type")
        cmdline.append("poisson")
        cmdline.append("--client-sampling-poisson")
        cmdline.append(configView.edtPoissonSampling.text())
    elif sampling_index_type == 2:
        cmdline.append("--client-sampling-type")
        cmdline.append("poisson-no-empty")
        cmdline.append("--client-sampling-poisson")
        cmdline.append(configView.edtPoissonSampling.text())

    cmdline.append("--global-lr")
    cmdline.append(configView.edtGlbInitLearningRate.text())
    cmdline.append("--global-optimiser")
    cmdline.append(configView.cbxGlobalOptimizer.currentText())
    cmdline.append("--global-weight-decay")
    cmdline.append(configView.edtGlobalWeightDecay.text())
    cmdline.append("--number-of-local-iters")
    cmdline.append(configView.edtLocalIterations.text())

    if configView.cbxLocalIterationsType.currentText() == "local-steps":
        cmdline.append("--run-local-steps")

    cmdline.append("--batch-size")
    cmdline.append(configView.edtDataLoadBatchSize.text())

    cmdline.append("--local-lr")
    cmdline.append(configView.edtLocalInitialLr.text())
    cmdline.append("--local-optimiser")
    cmdline.append(configView.cbxLocalOpt.currentText())
    cmdline.append("--local-weight-decay")
    cmdline.append(configView.edtLocalWeightDecay.text())

    ds = configView.cbxDataset.currentText().lower()
    cmdline.append("--dataset")
    cmdline.append(ds)

    # Dataset generation specification
    if ds == "generated_for_quadratic_minimization":
        dsGenSpec = []
        dsGenSpec.append(f"homogeneous:{int(configView.cbxHomogeneousDS.checkState() == Qt.Checked)}")
        dsGenSpec.append(f"mu:{str(configView.edtMu.text())}")
        dsGenSpec.append(f"L:{str(configView.edtL.text())}")
        dsGenSpec.append(f"samples_per_client:{str(configView.edtSamplesPerClient.text())}")
        dsGenSpec.append(f"clients:{str(configView.edtNumberOfClients.text())}")
        dsGenSpec.append(f"variables:{str(configView.edtVarsInOpt.text())}")
        dsGenSpecStr = ",".join(dsGenSpec)
        cmdline.append("--dataset-generation-spec")
        cmdline.append(dsGenSpecStr)

        cmdline.append("--loss")
        cmdline.append("mse")
        cmdline.append("--model")
        cmdline.append("linear")
        cmdline.append("--metric")
        cmdline.append("loss")

    else:
        cmdline.append("--loss")
        cmdline.append(configView.cbxLossFunctionForGlobalModel.currentText())
        cmdline.append("--model")
        cmdline.append(configView.cbxGlobalModel.currentText())

        if configView.cbxPretrained.checkState() == Qt.Checked:
            cmdline.append("--use-pretrained")

        if configView.cbxTrainLastLayerOnly.checkState() == Qt.Checked:
            cmdline.append("--train-last-layer")

        if configView.cbxTurnOffBatchNormAndDropOut.checkState() == Qt.Checked:
            cmdline.append("--turn-off-batch-normalization-and-dropout")

        cmdline.append("--metric")
        cmdline.append(configView.cbxValidationOptMetric.currentText())

    cmdline.append("--global-regulizer")
    cmdline.append(configView.cbxGlobalLossRegulizer.currentText())
    cmdline.append("--global-regulizer-alpha")
    cmdline.append(configView.edtGlobalRegulizerCoefficent.text())

    cmdline.append("--checkpoint-dir")
    cmdline.append(configView.edtCheckpointDir.text())

    if configView.cbxDoNotSaveEvalCheckpoints.checkState() == Qt.Checked:
        cmdline.append("--do-not-save-eval-checkpoints")

    cmdline.append("--data-path")
    cmdline.append(configView.edtDataPath.text())
    cmdline.append("--compute-type")
    cmdline.append(configView.cbxParamsType.currentText())
    cmdline.append("--gpu")
    device_index = configView.cbxComputeDevice.currentIndex()

    if device_index == 0:
        # Single CPU
        device_index = -1
        cmdline.append(str(device_index))
    elif device_index == configView.cbxComputeDevice.count() - 1:
        # Complicated selection - CPUs, GPUs, or even external compute resources
        devicesList = []

        # Process local devices
        gpus_properties = gpu_utils.get_available_gpus()
        for i in range(len(gpus_properties)):
            if configView.loComputeDevices.itemAt(i + 1).widget().checkState() == Qt.Checked:
                devicesList.append(i)

        if configView.loComputeDevices.itemAt(0).widget().checkState() == Qt.Checked:
            devicesList.append(-1)

        if len(devicesList) == 0:
            dlg = QMessageBox()
            dlg.setWindowTitle("Error")
            dlg.setText(f"You have to select at least one computation device in Configuration/System Aspects")
            dlg.setStandardButtons(QMessageBox.Ok)
            dlg.setIcon(QMessageBox.Critical)
            button = dlg.exec_()
            return

        devString = ",".join([str(d) for d in devicesList])
        cmdline.append(devString)

        # Cmdline option for external resources
        if configView.loComputeDevices.itemAt(configView.loComputeDevices.count() - 1).widget().checkState() == Qt.Checked:
            cmdline.append("--external-devices")
            usedMachines = []
            for m in machineDescriptions:
                for md in m['used_devices']:
                    usedMachines.append(m["ip"] + ":" + str(m["port"]) + ":" + md)
            machineString = ",".join([str(m) for m in usedMachines])
            cmdline.append(machineString)
    else:
        # Single GPU
        device_index = device_index - 1
        cmdline.append(str(device_index))

    if configView.cbxLogGPUusage.checkState() == Qt.Checked:
        cmdline.append("--log-gpu-usage")

    cmdline.append("--num-workers-train")
    cmdline.append(configView.edtNumWorkersForTrainDataLoad.text())

    cmdline.append("--num-workers-test")
    cmdline.append(configView.edtNumWorkersForValDataLoad.text())

    if configView.cbxRunDeterministically.checkState() == Qt.Checked:
        cmdline.append("--deterministic")

    cmdline.append("--manual-init-seed")
    cmdline.append(configView.edtRandomInitSeed.text())

    cmdline.append("--manual-runtime-seed")
    cmdline.append(configView.edtRandomRunSeed.text())

    cmdline.append("--group-name")
    cmdline.append(configView.edtGroupName.text())

    cmdline.append("--comment")
    cmdline.append(configView.edtCommentForExperiment.text())

    cmdline.append("--hostname")
    cmdline.append(socket.gethostname())

    cmdline.append("--eval-every")
    cmdline.append(configView.edtValidationAssementFreq.text())

    cmdline.append("--eval-async-threads")
    cmdline.append(configView.edtEvalThreadPool.text())
    cmdline.append("--save-async-threads")
    cmdline.append(configView.edtSerializingThreadPool.text())
    cmdline.append("--threadpool-for-local-opt")
    cmdline.append(configView.edtLocalTraininThreadPool.text())
    cmdline.append("--run-id")
    cmdline.append(configView.edtJobId.text())
    cmdline.append("--algorithm")
    cmdline.append(configView.cbxOptAlgo.currentText())
    cmdline.append("--algorithm-options")

    # Append experimental optimization options
    full_algo_opts = ""
    algo_opts = configView.cbxGradCalculation.currentText().lower().strip()
    algo_opts = cleanupFromComments(algo_opts)

    if algo_opts.rfind(' ') != -1:
        algo_opts = algo_opts[0 : algo_opts.rfind(' ')]

    full_algo_opts += "internal_sgd:" + algo_opts + ","
    full_algo_opts += configView.edtExperimentalExtraOpts.text().lower().strip().replace(";",",").replace("=",":")
    full_algo_opts = [a.strip() for a in full_algo_opts.split(",") if len(a.strip()) > 0]
    cmdline.append(",".join(full_algo_opts))

    cmdline.append("--logfile")
    cmdline.append(configView.edtLogFile.text())

    # Client compressor specification
    iSelectedCompressor = configView.cbxClientCompressor.currentIndex()
    compressorType = compressorTypes[iSelectedCompressor].lower().strip()
    compressorParams = configView.edtClientCompressorConfig.text().lower().strip()
    if compressorParams[0:len(compressorType) + 1] == compressorType + ":":
        compressorParams = compressorParams[len(compressorType) + 1:]
    compressorCmd = compressorType
    if len(compressorParams) > 0:
        compressorCmd += (":" + compressorParams)

    cmdline.append("--client-compressor")
    cmdline.append(compressorCmd)

    cmdline.append("--extra-track")
    extra_track = []
    if configView.cbxTrackNormFullGradientTrain.checkState() == Qt.Checked:
        extra_track.append("full_gradient_norm_train")
    if configView.cbxTrackObjectiveFunctionValueTrain.checkState() == Qt.Checked:
        extra_track.append("full_objective_value_train")
    if configView.cbxTrackNormFullGradientVal.checkState() == Qt.Checked:
        extra_track.append("full_gradient_norm_val")
    if configView.cbxTrackObjectiveFunctionValueVal.checkState() == Qt.Checked:
        extra_track.append("full_objective_value_val")
    extra_track_str = ",".join(extra_track)
    cmdline.append(extra_track_str)

    if configView.cbxStoreClientStateInCPU.checkState() == Qt.Checked:
        cmdline.append("--store-client-state-in-cpu")

    if configView.cbxEmptyGPUTorchCache.checkState() == Qt.Checked:
        cmdline.append("--per-round-clean-torch-cache")

    if configView.cbxAllowUseNvTensorCores.checkState() == Qt.Checked:
        cmdline.append("--allow-use-nv-tensorcores")

    cmdline.append("--initialize-shifts-policy")
    shift_policy = configView.cbxInitializeShiftsPolicy.currentText().lower()
    cmdline.append(shift_policy)

    if configView.cbxSortDatasetByClassBeforeSplit.checkState() == Qt.Checked:
        cmdline.append("--sort-dataset-by-class-before-split")

    # append wandb key
    cmdline.append("--wandb-key")
    cmdline.append(configView.edtWandbKey.text().strip())

    # append wandb project
    cmdline.append("--wandb-project-name")
    cmdline.append(configView.edtWandbProjectName.text().strip())

    # Lowering all letters in command and strip whitespaces and make substitutions:
    #
    #   now - number of second since unix epoch
    #   compressor - type of used compressor
    #   algorithm - optimization algorithm
    # All this constants can be used in parameters for purpose of substitution

    #now = int(time.time())
    #for i in range(len(cmdline)):
    #    cmdline[i] = str(cmdline[i]).lower().strip().format(now = now, compressor = compressorType, algorithm = configView.cbxOptAlgo.currentText(), simcounter = gSimulationCounter)

    # Append commands which are case sensitive
    cmdline.append("--loglevel")
    cmdline.append(configView.cbxLoggingLevels.currentText())

    cmdline.append("--logfilter")
    cmdline.append(configView.edtLoggingFilter.text().strip())

    return cmdline

def onStartSimulation():
    """Initiate and dispatch simulation"""
    cmdline = currentCommandLineFromGUI()

    #cmdline = "run.py --run-id test --rounds 25 --num-clients-per-round 8 -b 32 --dataset cifar100_fl --model resnet18 --deterministic -li 1 --eval-async-threads 1 --threadpool-for-local-opt 8"
    simulationThreadPool.dispatch(lambda th, cmd: run.runSimulation(cmd), (cmdline,), simulationThreadPool.get_free_worker_index())

    # Notify about launching simulation job
    job_id = ""
    for i in range(len(cmdline)):
        if cmdline[i] == "--run-id":
            job_id = cmdline[i + 1]

    msg = f"Job '{job_id}' with algorithm '{configView.cbxOptAlgo.currentText()}' has been sumbitted"

    mainView.statusBar.showMessage(msg, 1000)
    uiLogInfo(msg)


def onCancelAllSimulations():
    global earlyStopSimulations, simulations_stats, sim_stats_res_lock

    # Register for early stopping all jobs
    sim_stats_res_lock.acquire()
    earlyStopSimulations = set()
    for job_id in simulations_stats.keys():
        if not simulations_stats[job_id]["finished"]:
            earlyStopSimulations.add(job_id)
    sim_stats_res_lock.release()
    uiLogInfo(f"Request to stop jobs: {earlyStopSimulations}")


def onCancelSimulation():
    global earlyStopSimulations, simulations_stats, simulations_by_rows

    sim_stats_res_lock.acquire()

    selected_items = simulationView.tblExperiments.selectedItems()
    rows = {item.row() for item in selected_items}
    for row in rows:
        job_id = simulations_by_rows[row]
        is_finished = simulations_stats[job_id]["finished"]
        if not is_finished:
            earlyStopSimulations.add(job_id)

    sim_stats_res_lock.release()
    uiLogInfo(f"Request to stop jobs: {earlyStopSimulations}")


def onSelectDataset(index):
    dataset = configView.cbxDataset.itemText(index).lower()
    number_of_clients = ""

    if dataset == "generated_for_quadratic_minimization":
        number_of_clients = "10"
        showArtificialDatasetParameters(True)
    else:
        try:
            number_of_clients = str(data_preprocess.data_loader.get_num_clients(dataset))
        except:
            showError("Failed to get number of clients for dataset")
            pass

        showArtificialDatasetParameters(False)

    configView.edtNumberOfClients.setText(number_of_clients)


def onCleanCheckpointAndLogs():
    dlg = QMessageBox()
    # Change the title of our dialog window
    dlg.setWindowTitle("Confirmation about cleaning folders")
    checkpoint = configView.edtCheckpointDir.text()
    dlg.setText(f"Are you sure you want to clean '{checkpoint}' ?")
    dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    dlg.setIcon(QMessageBox.Question)
    button = dlg.exec_()
    if button != QMessageBox.Yes:
        return

    if os.path.isdir(checkpoint):
        shutil.rmtree(checkpoint)
        uiLogInfo(f"Folder '{checkpoint}' has been removed")
    else:
        uiLogInfo(f"There is no folder '{checkpoint}' in a filesystem")


def onChangeSelectedExperimentsInSimView():
    selected_items = simulationView.tblExperiments.selectedItems()
    rows = {item.row() for item in selected_items}
    number_of_selected_experiments = len(rows)
    msgText = f"The {number_of_selected_experiments}/{len(simulations_by_rows)} of available experiments have been selected."
    mainView.statusBar.showMessage(msgText, 2000)


def onChangeSelectedExperimentsInAnalysisView():
    selected_items = analysisView.tblExperiments.selectedItems()
    rows = {item.row() for item in selected_items}
    number_of_selected_experiments = len(rows)
    msgText = f"The {number_of_selected_experiments}/{len(simulations_by_rows)} of available experiments have been selected."
    mainView.statusBar.showMessage(msgText, 2000)


def onLoadSummaryTable():
    selected_items = simulationView.tblExperiments.selectedItems()
    selected_row = simulationView.tblExperiments.currentRow()

    if len(selected_items) == 0 or selected_row < 0:
        while simulationView.tblSummary.rowCount() > 0:
            simulationView.tblSummary.removeRow(0)
        return

    if selected_row >= len(simulations_by_rows):
        return

    job_id = simulations_by_rows[selected_row]
    stats = simulations_stats[job_id]
    # Dictionary with filtered statistics
    filtered_stats = {}

    # ===================================================================================================================
    # Apply filter in form "substring1|substring2|substring 3" and exclude server state from loading into result table
    filter = simulationView.edtFieldsThatContains.text().strip().split("|")

    if len(filter) == 1 and filter[0] == "":
        sim_stats_res_lock.acquire()
        for k,v in stats.items():
            if k != 'H':
                filtered_stats[k] = v
        sim_stats_res_lock.release()
    else:
        sim_stats_res_lock.acquire()

        for k,v in stats.items():
            if k == 'H':
                continue

            for curFilter in filter:
                if curFilter == "":
                    continue
                if k.find(curFilter) >= 0 or str(v).find(curFilter) >= 0:
                    filtered_stats[k] = v

        sim_stats_res_lock.release()
    # ==================================================================================================================
    w = simulationView.tblSummary
    while w.rowCount() < len(filtered_stats):
        it_prop = QtWidgets.QTableWidgetItem("")
        it_value = QtWidgets.QTableWidgetItem("")
        w.insertRow(w.rowCount())
        r = w.rowCount() - 1
        for c, item in enumerate((it_prop, it_value)):
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            w.setItem(r, c, item)
    while w.rowCount() > len(filtered_stats):
        w.removeRow(w.rowCount() - 1)

    for i, (prop, value) in enumerate(filtered_stats.items()):
        w.item(i, 0).setText(str(prop))
        w.item(i, 1).setText(str(value))


def onBtnCleanExperimentsColumns():
    analysisView.tblExperiments.setColumnCount(1)
    onUpdateTableForAnalysis()

    header = analysisView.tblExperiments.horizontalHeader()
    for i in range(len(header) - 1):
        header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
    header.setSectionResizeMode(len(header) - 1, QtWidgets.QHeaderView.Stretch)


def onBtnAddExperimentsColumn():
    analysisView.tblExperiments.setColumnCount(1 + analysisView.tblExperiments.columnCount())
    header = analysisView.tblExperiments.horizontalHeader()

    analysisView.tblExperiments.setHorizontalHeaderItem(len(header) - 1,  QtWidgets.QTableWidgetItem())
    analysisView.tblExperiments.horizontalHeaderItem(len(header) - 1).setText(analysisView.cbxbAddExperimentColumn.currentText())

    onUpdateTableForAnalysis()

    header = analysisView.tblExperiments.horizontalHeader()
    for i in range(len(header) - 1):
        header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
    header.setSectionResizeMode(len(header) - 1, QtWidgets.QHeaderView.Stretch)


def onRemoveSelectedExperiment():
    global  simulations_stats, simulations_result, simulations_by_rows

    selected_items = simulationView.tblExperiments.selectedItems()
    rows = {item.row() for item in selected_items}

    if len(selected_items) == 0:
        return

    job_ids = set()
    job_id_str = ""

    for row in rows:
        job_id = simulations_by_rows[row]
        job_ids.update({job_id})
        job_id_str = job_id_str + " " + job_id

        is_finished = simulations_stats[job_id]["finished"]

        if not is_finished:
            dlg = QMessageBox()
            dlg.setWindowTitle("Impossible to remove experiment")
            dlg.setText(f"Experiment {job_id} is currently running. Please stop it first.")
            dlg.setStandardButtons(QMessageBox.Ok)
            dlg.setIcon(QMessageBox.Information)
            button = dlg.exec_()
            return

    dlg = QMessageBox()
    dlg.setWindowTitle("Confirmation about experimental results")
    dlg.setText(f"Are you sure you want to drop experimental results for {job_id_str}?")
    dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    dlg.setIcon(QMessageBox.Question)
    button = dlg.exec_()
    if button != QMessageBox.Yes:
        return

    # ==================================================================================================================
    sim_stats_res_lock.acquire()

    for del_job in job_ids:
        del simulations_result[del_job]
        del simulations_stats[del_job]

    i = len(simulations_by_rows) - 1
    while i >= 0:
        if simulations_by_rows[i] in job_ids:
            del simulations_by_rows[i]

        i -= 1

    # ==================================================================================================================
    assert len(simulations_by_rows) == len(simulations_stats)
    assert len(simulations_by_rows) >= len(simulations_result)
    # ==================================================================================================================

    sim_stats_res_lock.release()


def onRemoveAllExperiments():
    total_commands = totalSimulations()

    if total_commands != 0:
        dlg = QMessageBox()
        dlg.setWindowTitle("Confirmation about removing all experimental results")
        dlg.setText(f"There are {total_commands} current execution. Please stop them first.")
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setIcon(QMessageBox.Information)
        dlg.exec_()
        return
    else:
        dlg = QMessageBox()
        dlg.setWindowTitle("Confirmation about removing all experimental results")
        dlg.setText(f"Are you sure you want to drop all experimental results?")
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dlg.setIcon(QMessageBox.Question)
        button = dlg.exec_()
        if button != QMessageBox.Yes:
            return

        global  simulations_stats, simulations_result, simulations_by_rows
        # ==========================================================
        sim_stats_res_lock.acquire()
        simulations_stats = {}    # Simulations key - job_id, value collected statistics
        simulations_result = {}   # Simulation result, key - job_id, value server-state (stop to change)
        simulations_by_rows = []  # Simulations, index - row in table, value - job_id
        sim_stats_res_lock.release()
        # ==========================================================
        while simulationView.tblSummary.rowCount() > 0:
            simulationView.tblSummary.removeRow(0)
        # ==========================================================


def sortSimulationsByStatsKey(simulationKeyForSorting):
    global simulations_by_rows, simulations_stats, simulations_result, sim_stats_res_lock

    if simulationKeyForSorting == None:
        return

    simulations_by_rows_sorted = []
    algo2index = {}
    for index, a in enumerate(algorithms.getAlgorithmsList()):
        algo2index[a] = index

    sim_stats_res_lock.acquire()

    # Collect (compare_key, jobId) from (jobId) list
    for job_id in simulations_by_rows:
        if simulationKeyForSorting not in simulations_stats[job_id]:
            sim_stats_res_lock.release()
            msgText = f"Sorting simulation by '{simulationKeyForSorting}' has been FAILED!"
            mainView.statusBar.showMessage(msgText, 1500)
            uiLogInfo(msgText)
            return

        # Obtain value which we will use as a sorting criteria
        jobStatValue = simulations_stats[job_id][simulationKeyForSorting]

        # Ad-hoc for several sorting
        if simulationKeyForSorting == "algorithm":
            if jobStatValue in algo2index:
                jobStatValue = algo2index[jobStatValue]
            else:
                jobStatValue = 100

        simulations_by_rows_sorted.append( (jobStatValue, job_id) )

    # Sort auxilirary list
    simulations_by_rows_sorted.sort()

    # Unpack sorted list with throw away sorting key
    simulations_by_rows = [element[1] for element in simulations_by_rows_sorted]
    sim_stats_res_lock.release()

    msgText = f"Sorting simulation by '{simulationKeyForSorting}' has been finished"
    mainView.statusBar.showMessage(msgText, 1500)
    uiLogInfo(msgText)

    # Update table
    onUpdateTableForAnalysis()


def onLoadExperimentResults():
    global simulations_by_rows, simulations_stats, simulations_result

    fileNames,_ = QFileDialog.getOpenFileNames(mainView,
                                              "Append experimental results serialized in binary format",
                                              "",
                                              "Binary file (*.bin)"
                                              )
    if len(fileNames) == 0:
        return

    simulation_keys = []

    # ==================================================================================================================
    for fileName in fileNames:
        if len(fileName) == 0:
            continue

        obj = None

        try:
            with open(fileName, "rb") as f:
                obj = pickle.load(f)
        except:
            showError(f"Failed to load experiments from: '{fileName}'")
            return

        list_job_ids             = [jobId  for jobId, result in obj]
        simulations_result_extra = {jobId:result for jobId, result in obj}

        list_of_ignored_job_ids = []
        list_of_loaded_job_ids = []
        # ==========================================================
        for job_id in list_job_ids:
            if configView.cbxLoadNameMangling.checkState() == Qt.Checked:
                job_id_new = os.path.basename(fileName).replace(".bin", "") + "_" + job_id + "_L_" + str(int(time.time()))
            else:
                job_id_new = job_id

            if job_id in simulations_stats:
                list_of_ignored_job_ids.append(job_id)
            else:
                list_of_loaded_job_ids.append(job_id)

                H = simulations_result_extra[job_id]
                H["run_id"] = job_id_new

                sim_stats_res_lock.acquire()
                simulations_stats[job_id_new] = {}
                simulations_stats[job_id_new]["source"] = fileName

                simulations_by_rows.append(job_id_new)
                simulationFinishInternal(H, lock = False)
                sim_stats_res_lock.release()
        # ==========================================================
        simulation_keys += list(simulations_result_extra.keys())
    # ==================================================================================================================

    uiLogInfo(f"#{len(list_of_loaded_job_ids)} simulations {list_of_loaded_job_ids} have been successfully loaded from: {fileNames}")
    uiLogInfo(f"#{len(list_of_ignored_job_ids)} simulations {list_of_ignored_job_ids} have been ignored from: {fileNames}")

    if len(list_of_loaded_job_ids) > 0:
        dlg = QMessageBox()
        dlg.setWindowTitle("The jobs have been loaded successfully")
        dlg.setText(f"#{len(list_of_loaded_job_ids)} simulations \n {list_of_loaded_job_ids} \n have been successfully loaded from \n {fileNames}.")
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setIcon(QMessageBox.Information)
        button = dlg.exec_()

    if len(list_of_ignored_job_ids) > 0:
        dlg = QMessageBox()
        dlg.setWindowTitle("Several jobs have not been loaded")
        dlg.setText(f"#{len(list_of_ignored_job_ids)} simulations \n {list_of_ignored_job_ids} \n have been ignored from \n {fileNames}!")
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setIcon(QMessageBox.Warning)
        button = dlg.exec_()

    # sortSimulationsByStatsKey("algorithm")


def onSaveExperimentResults(selected):
    if selected:
        windowTitle = "Save selected and finished experimental results in serialized binary format"
    else:
        windowTitle = "Save all finished experimental results in serialized binary format"

    fileName, _ = QFileDialog.getSaveFileName(mainView, windowTitle, "", "Binary file (*.bin)")

    if len(fileName) == 0:
        return

    if not fileName.endswith(".bin"):
        fileName = fileName + ".bin"

    selectedRows = set()

    if mainView.tabMainView.currentIndex() == 1:
        # Selected rows in a widget for simulation
        selectedRows = {item.row() for item in simulationView.tblExperiments.selectedItems()}
    elif mainView.tabMainView.currentIndex() == 2:
        # Selected rows in a widget for analysis
        selectedRows = {item.row() for item in analysisView.tblExperiments.selectedItems()}
    # ==================================================================================================================

    simulations_result_2_save = []

    # ==================================================================================================================
    sim_stats_res_lock.acquire()
    for row in range(len(simulations_by_rows)):
        if selected:
            if row not in selectedRows:
                # If we are saving only selected items and items has not been selected - ingnore experiment
                continue

        job_id = simulations_by_rows[row]
        if simulations_stats[job_id]["finished"]:
            simulations_result_2_save.append( (job_id, simulations_result[job_id]) )
    sim_stats_res_lock.release()
    # ==================================================================================================================
    try:
        obj = simulations_result_2_save
        with open(fileName, "wb") as f:
            pickle.dump(obj, f)
    except:
        showError(f"Failed to save experiments into: '{fileName}'")
        return

    simulations = [jobId for jobId, result in simulations_result_2_save]

    # Update source path for experiments. start
    sim_stats_res_lock.acquire()
    for jobId, result in simulations_result_2_save:
        simulations_stats[jobId]["source"] = fileName
    sim_stats_res_lock.release()
    # Update source path for experiments. end

    uiLogInfo(f"#{len(simulations)} simulations {simulations} have been successfully serialized into: {fileName}")
    dlg = QMessageBox()
    dlg.setWindowTitle("Confirmation")
    dlg.setText(f"#{len(simulations)} simulations \n {simulations} have been successfully serialized into:\n {fileName}")
    dlg.setStandardButtons(QMessageBox.Ok)
    dlg.setIcon(QMessageBox.Information)
    button = dlg.exec_()

def onHeaderAnalysisExperimentClicked(logicalIndex):
    key = analysisView.tblExperiments.horizontalHeaderItem(logicalIndex).text().lower()

    #if key == "job":
    #    key = "run_id"
    #elif key == "d":
    #    key = "D"
    #elif key == "exper.group":
    #    key = "group_name"
    #elif key == "exper.config":
    #    key = None
    #elif key == "compressor":
    #    key = "client_compressor"

    sortSimulationsByStatsKey(key)


def cleanPlot(plt, tab_index):
    ax = plt.axes
    fig = plt.fig
    title_text = ax.title.get_text()

    ax.clear()
    # For debug
    # ax.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
    ax.grid(True)

    if title_text == "":
        ax.set_title(analysisView.tabWidget.tabText(tab_index))
    else:
        ax.set_title(title_text)

    plt.draw()


def showError(msg):
    dlg = QMessageBox()
    dlg.setWindowTitle("Error")
    dlg.setText(msg)
    dlg.setStandardButtons(QMessageBox.Ok)
    dlg.setIcon(QMessageBox.Critical)
    button = dlg.exec_()
    return


def toAxisName(axis):
    """Specialized translation of axis names for more nice names for axis names in a final plots"""
    if axis == "Norm of function gradient square(train)":
       return "${||\\nabla F(x)||}^2$ (train)"

    if axis == "Norm of function gradient (train)":
       return "${||\\nabla F(x)||}$ (train)"

    if axis == "Norm of function gradient square(validation)":
       return "$||\\nabla F(x)||^2$ (validation)"

    if axis == "Norm of function gradient (validation)":
       return "${||\\nabla F(x)||}$ (validation)"

    if axis == "Function value (train)":
       return "$F(x)$ (train)"

    if axis == "Function value (validation)":
       return "$F(x)$ (validation)"

    if axis == "# of bits from clients/#workers":
        return "#bits/$n_i$"

    if axis == "# of bits from clients/#total workers":
        return "#bits/n"

    if axis == "# of bits from clients":
        return "#bits"

    if axis == "Distance to solution squared (train)":
       return "${||x - x^*||^2}$ (train)"

    return axis


def moveSimulationUp():
    sim_stats_res_lock.acquire()
    selected_items = analysisView.tblExperiments.selectedItems()
    rows = {item.row() for item in selected_items}
    rows = sorted(list(rows))
    for r in rows:
        if r == 0:
            continue

        tmp = simulations_by_rows[r-1]
        simulations_by_rows[r-1] = simulations_by_rows[r]
        simulations_by_rows[r] = tmp

    for r in range(analysisView.tblExperiments.rowCount()):
        for c in range(analysisView.tblExperiments.columnCount()):
            item = analysisView.tblExperiments.item(r, c)
            isSelected = (r + 1 in rows)
            item.setSelected(isSelected)

    sim_stats_res_lock.release()
    onUpdateTableForAnalysis()


def moveSimulationDown():
    sim_stats_res_lock.acquire()
    selected_items = analysisView.tblExperiments.selectedItems()
    rows = {item.row() for item in selected_items}
    rows = sorted(list(rows))
    for r in reversed(rows):
        if r == len(simulations_by_rows) - 1:
            continue

        tmp = simulations_by_rows[r+1]
        simulations_by_rows[r+1] = simulations_by_rows[r]
        simulations_by_rows[r] = tmp

    for r in range(analysisView.tblExperiments.rowCount()):
        for c in range(analysisView.tblExperiments.columnCount()):
            item = analysisView.tblExperiments.item(r, c)
            isSelected = (r - 1 in rows)
            item.setSelected(isSelected)

    sim_stats_res_lock.release()
    onUpdateTableForAnalysis()


def includeToLegend():
    selText = analysisView.cbxIncludeToLegend.currentText()
    add_to_legend = ""
    if selText == "job id":
        add_to_legend = "{job_id}"
    elif selText == "compressors schema":
        add_to_legend = "{client_compressor}"
    elif selText == "algorithm name":
        add_to_legend = "{algorithm}"
    elif selText == "algorithm options":
        add_to_legend = "{algorithm_options}"
    elif selText == "number of local iterations":
        add_to_legend = "{local_iterations}"
    elif selText == "global step size":
        add_to_legend = "{global_lr}"
    elif selText == "local step size":
        add_to_legend = "{local_lr}"
    elif selText == "comment":
        add_to_legend = "{comment}"
    elif selText == "group name":
        add_to_legend = "{group_name}"
    elif selText == "total rounds":
        add_to_legend = "{rounds}"
    elif selText == "completed rounds":
        add_to_legend = "{completed_rounds}"
    elif selText == "hostname":
        add_to_legend = "{hostname}"
    elif selText == "dimension of x":
        add_to_legend = "{D}"
    elif selText == "total number of clients":
        add_to_legend = "{total_clients}"
    elif selText == "start time for simulation":
        add_to_legend = "{start_time}"
    elif selText == "aggregated experiments":
        add_to_legend = "{aggregated}"

    analysisView.edtLegendFormatString.setText(analysisView.edtLegendFormatString.text() + add_to_legend)


def onPlotResults(export_legend_to_file):
    customizeMatplotLib()

    plt_cur = analysisView.customPlots[ analysisView.tabWidget.currentIndex()]
    selected_items = analysisView.tblExperiments.selectedItems()
    rows = {item.row() for item in selected_items}
    job_ids = [simulations_by_rows[i] for i in sorted(list(rows))]

    if len(job_ids) == 0:
        cleanPlot(plt_cur, analysisView.tabWidget.currentIndex())
        return

    ax = plt_cur.axes
    fig = plt_cur.fig
    ax.clear()

    # Set font for extra text in axis
    extraAxisTextFont = int(0.8 * analysisView.sbxFontSizes.value())
    ax.xaxis.get_offset_text().set_fontsize(extraAxisTextFont)
    ax.yaxis.get_offset_text().set_fontsize(extraAxisTextFont)

    # Built-in markers
    marker = ["x","^","*","x","^","*"]
    # gui_rnd.shuffle(marker)

    linestyle = ["dashed", "solid", "dashed", "solid", "dashed", "solid"]

    if analysisView.cbxLineStyle.currentIndex() == 0:
        linestyle = ["solid"] * len(job_ids)
    elif analysisView.cbxLineStyle.currentIndex() == 1:
        linestyle = ["dashed"] * len(job_ids)
    elif analysisView.cbxLineStyle.currentIndex() == 2:
        linestyle = ["solid"] * (len(job_ids) // 2) + ["dashed"] * (len(job_ids) // 2)
        if len(job_ids) % 2 == 1: linestyle.append("dashed")
    elif analysisView.cbxLineStyle.currentIndex() == 3:
        linestyle = ["dashed"] * (len(job_ids) // 2) + ["solid"] * (len(job_ids) // 2)
        if len(job_ids) % 2 == 1: linestyle.append("solid")

    # https://matplotlib.org/stable/gallery/color/named_colors.html
    color = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    # Hack
    #color = color[1:]
    #marker = marker[1:]

    fontSize = analysisView.sbxFontSizes.value()

    plots = []

    # Plot current results if plotting has been required been done during experiments
    sim_stats_res_lock.acquire()
    for g, job_id in enumerate(job_ids):
        uiLogInfo(f"Plot results for experiment {job_id}")

        # Server state for simulation
        H = simulations_stats[job_id]['H']

        # Pull information about completed rounds
        rounds = simulations_stats[job_id]["completed_rounds"]

        X = [0.0] * rounds
        Y = [0.0] * rounds

        # =================================================================================================================
        xAxisType = str(analysisView.cbxXaxis.currentText())
        yAxisType = str(analysisView.cbxYaxis.currentText())

        full_gradient_oracles   = [0.0] * rounds
        sample_gradient_oracles = [0.0] * rounds
        send_scalars_to_master  = [0.0] * rounds
        send_scalars_to_master_div_clients = [0.0] * rounds

        # ==============================================================================================================
        # Amount of bits in one scalar - by default we assume that type of scalar component x[i] in trainable vector of parameters "x" is fp32
        one_scalar_in_bits = 32

        # Compute aggregated statistics
        for round in range(rounds):
            round_info = H['history'][round]
            num_clients_in_round = len(round_info['client_states'].keys())

            for client_id, client_summary in round_info['client_states'].items():
                client_stats = client_summary['client_state']['stats']

                send_scalars_to_master[round] += float(client_stats['send_scalars_to_master'])
                send_scalars_to_master_div_clients[round] += float(client_stats['send_scalars_to_master']) / float(num_clients_in_round)

                full_gradient_oracles[round] += float(client_stats['full_gradient_oracles'])
                sample_gradient_oracles[round] += float(client_stats['samples_gradient_oracles'])


        for i in range(1,rounds):
            full_gradient_oracles[i]    += full_gradient_oracles[i-1]
            sample_gradient_oracles[i]  += sample_gradient_oracles[i-1]
            send_scalars_to_master[i]   += send_scalars_to_master[i-1]
            send_scalars_to_master_div_clients[i] += send_scalars_to_master_div_clients[i-1]

        rounds_with_actual_info = 0
        stop_round_traversing = False

        # Traverse across completed rounds and collect what we can
        for round in range(rounds):
            round_info = H['history'][round]

            for axis_type, axis_values in ( (xAxisType, X), (yAxisType, Y) ):
                if axis_type == "Rounds":
                    axis_values[round] = round

                if axis_type == "Time (minutes)":
                    field = "time"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field] / 60.0

                if axis_type == "Time per one communication round (minutes)":
                    field = "train_time"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field] / 60.0

                if axis_type == "Norm of server stochastic gradient (proxy train)":
                    field = "grad_sgd_server_l2"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field]

                if axis_type == "Running function value (proxy train)":
                    field = "approximate_f_avg_value"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field]

                if axis_type == "Norm of function gradient (train)":
                    field = "full_gradient_norm_train"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field]

                if axis_type == "Norm of function gradient square(train)":
                    field = "full_gradient_norm_train"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field]**2


                if axis_type == "Function value (train)":
                    field = "full_objective_value_train"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field]

                if axis_type == "Norm of iterate at round start (train)":
                    field = "x_before_round"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field]

                if axis_type == "Distance to solution squared (train)":
                    field = "distance_to_solution"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info["distance_to_solution"]**2

                if axis_type == "Norm of function gradient square(validation)":
                    field = "full_gradient_norm_val"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field]**2

                if axis_type == "Norm of function gradient (validation)":
                    field = "full_gradient_norm_val"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field]

                if axis_type == "Function value (validation)":
                    field = "full_objective_value_val"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = round_info[field]

                if axis_type == "Number of compute devices":
                    field = "number_of_client_in_round"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = int(round_info[field])

                if axis_type == "GPU Memory (Megabytes)":
                    field = "memory_gpu_used"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = float(round_info[field])

                if axis_type == "GPU Memory (Gigabytes)":
                    field = "memory_gpu_used"
                    if field not in round_info:
                        stop_round_traversing = True
                        break
                    axis_values[round] = float(round_info[field]) / 1024.0

                if axis_type == "# of bits from clients/#workers":
                    axis_values[round] = send_scalars_to_master_div_clients[round] * one_scalar_in_bits

                if axis_type == "# of bits from clients/#total workers":
                    axis_values[round] = send_scalars_to_master[round] / H["total_clients"] * one_scalar_in_bits

                if axis_type == "# of bits from clients":
                    axis_values[round] = send_scalars_to_master[round] * one_scalar_in_bits

                if axis_type == "Full gradient oracles (train)":
                    axis_values[round] = full_gradient_oracles[round]

                if axis_type == "Sample gradient oracles (train)":
                    axis_values[round] = sample_gradient_oracles[round]

                if axis_type == "Loss (validation)":
                    field = 'eval_metrics'
                    sub_field = 'loss'

                    if field not in H:
                        stop_round_traversing = True
                        break

                    if len(H['eval_metrics']) == 0:
                        axis_values[round] = 0.0
                    elif round in H['eval_metrics']:
                        axis_values[round] = H['eval_metrics'][round][sub_field]
                    else:
                        lower_r, upper_r = round, round
                        while lower_r not in H['eval_metrics']:
                            lower_r -= 1

                        while upper_r not in H['eval_metrics'] and upper_r < rounds:
                            upper_r += 1

                        if upper_r == rounds:
                            stop_round_traversing = True
                            break

                        alpha = (round - lower_r) / float(upper_r - lower_r)
                        axis_values[round] = H['eval_metrics'][lower_r][sub_field] * (1-alpha) + H['eval_metrics'][lower_r][sub_field] * alpha

                if axis_type == "Top1 acc (validation)":
                    field = 'eval_metrics'
                    sub_field = 'top_1_acc'

                    if field not in H:
                        stop_round_traversing = True
                        break

                    if len(H['eval_metrics']) == 0:
                        axis_values[round] = 0.0
                    elif round in H['eval_metrics']:
                        axis_values[round] = H['eval_metrics'][round][sub_field]
                    else:
                        lower_r, upper_r = round, round
                        while lower_r not in H['eval_metrics']:
                            lower_r -= 1

                        while upper_r not in H['eval_metrics'] and upper_r < rounds:
                            upper_r += 1

                        if upper_r == rounds:
                            stop_round_traversing = True
                            break

                        alpha = (round - lower_r) / float(upper_r - lower_r)
                        axis_values[round] = H['eval_metrics'][lower_r][sub_field] * (1-alpha) + H['eval_metrics'][lower_r][sub_field] * alpha

                if axis_type == "Top5 acc (validation)":
                    field = 'eval_metrics'
                    sub_field = 'top_5_acc'

                    if field not in H:
                        stop_round_traversing = True
                        break

                    if len(H['eval_metrics']) == 0:
                        axis_values[round] = 0.0
                    elif round in H['eval_metrics']:
                        axis_values[round] = H['eval_metrics'][round][sub_field]
                    else:
                        lower_r, upper_r = round, round
                        while lower_r not in H['eval_metrics']:
                            lower_r -= 1

                        while upper_r not in H['eval_metrics'] and upper_r < rounds:
                            upper_r += 1

                        if upper_r == rounds:
                            stop_round_traversing = True
                            break

                        alpha = (round - lower_r) / float(upper_r - lower_r)
                        axis_values[round] = H['eval_metrics'][lower_r][sub_field] * (1-alpha) + H['eval_metrics'][lower_r][sub_field] * alpha

            if stop_round_traversing:
                break

            rounds_with_actual_info += 1
        # =================================================================================================================
        X = X[0:rounds_with_actual_info]
        Y = Y[0:rounds_with_actual_info]
        # =================================================================================================================
        plot_args = {"X": X, "Y": Y, "group": "", "aggregated": 1}
        plot_args["job_id"] = job_id
        plot_args["client_compressor"] = H['client_compressor'].upper()
        plot_args["algorithm"] = H['args'].algorithm.upper()
        plot_args["algorithm_options"] = H['args'].algorithm_options.upper()
        plot_args["local_iterations"] = str(H['args'].number_of_local_iters)
        plot_args["global_lr"] = str(H['args'].global_lr)
        plot_args["local_lr"] = str(H['args'].local_lr)
        plot_args["comment"] = H["args"].comment
        plot_args["group_name"] = H["args"].group_name
        plot_args["rounds"] = H['args'].rounds
        plot_args["hostname"] = H["args"].hostname
        plot_args["completed_rounds"] = len(H['history'].keys())

        # =================================================================================================================
        # Copy printable things from server state
        # =================================================================================================================
        for key, value in H.items():
            if type(value) == str or type(value) == int or type(value) == float or type(value) == np.float64 or type(value) == np.float64:
                if key not in plot_args:
                    plot_args[key] = value
        # =================================================================================================================
        # Extra experimental options
        # =================================================================================================================
        # Setup extra options for experiments
        for option in H['args'].algorithm_options.split(","):
            kv = option.split(":")
            if len(kv) == 1:
                plot_args["opt-" + kv[0]] = True
            else:
                plot_args["opt-" + kv[0]] = kv[1]

        for option in H['args'].dataset_generation_spec.split(","):
            kv = option.split(":")
            if len(kv) == 1:
                plot_args["gen-" + kv[0]] = True
            else:
                plot_args["gen-" + kv[0]] = kv[1]
        # =================================================================================================================
        markers_numbers = [9, 10, 11, 12, 13, 15, 16, 17, 19]
        markers_count_selection = markers_numbers[gui_rnd.randint(0, len(markers_numbers)-1)]
        mark_every = math.ceil( len(X)/float(markers_count_selection))

        plot_args["markevery"] = mark_every
        plot_args["marker"] = marker[g % len(marker)]
        plot_args["linestyle"] = linestyle[g % len(linestyle)]

        ax.set_xlabel(toAxisName(xAxisType), fontdict = {'fontsize' : fontSize} )
        ax.set_ylabel(toAxisName(yAxisType), fontdict = {'fontsize' : fontSize} )

        # Optionally turn off markers
        if analysisView.cbxDrawMarkers.checkState() != Qt.Checked:
            plot_args["marker"] = None
            plot_args["markevery"] = None

        if "group-name" in H:
            plot_args["group"] = H["group-name"]

        plots.append(plot_args)
    sim_stats_res_lock.release()
    # ===================================================================================================================================
    # Usual plots
    # ===================================================================================================================================
    rasterizedPlots = 0
    doNotAggregate = (analysisView.cbxAggregateCurves.currentText() == "do not aggregate curves")

    if analysisView.cbxLegendLogScalingY.checkState() == Qt.Checked and doNotAggregate:
        for p in plots:
            try:
                formatedLabel = analysisView.edtLegendFormatString.text().format(**p)
            except:
                showError("The legend contains unknown substitution variables")
                return

            ax.semilogy(p["X"], p["Y"], marker = p["marker"], markevery = p["markevery"], linestyle = p["linestyle"], label = formatedLabel, color = color[rasterizedPlots % len(color)])
            rasterizedPlots += 1

    if analysisView.cbxLegendLogScalingY.checkState() != Qt.Checked and doNotAggregate:
        for p in plots:
            try:
                formatedLabel = analysisView.edtLegendFormatString.text().format(**p)
            except:
                showError("The legend contains unknown substitution variables")
                return

            ax.plot(p["X"], p["Y"],
                    marker = p["marker"],
                    markevery = p["markevery"],
                    linestyle = p["linestyle"],
                    label = formatedLabel,
                    color = color[rasterizedPlots % len(color)])

            rasterizedPlots += 1

    # ===================================================================================================================================
    # Plots with pre-aggregation
    # ===================================================================================================================================
    plot_error_bar = (analysisView.cbxDrawErrorBars.checkState() == Qt.Checked)       # Plot or not error bar

    if not doNotAggregate:
        groups_my = {}        # Estimation of expectation for Y quantity
        groups_mysqr = {}        # Estimation of second moment for Y quantity
        groups_n = {}         # Number of elements(curves) in group
        groups_and_styles = {}   # Plot style per group
        groups_order = []        # Order in which groups are appearing

        for p in plots:
            group = None
            if analysisView.cbxAggregateCurves.currentText() == "do not aggregate curves":
                group = None
            elif analysisView.cbxAggregateCurves.currentText() == "aggregate curves by experiment group":
                group = p["group"]
            elif analysisView.cbxAggregateCurves.currentText() == "aggregate curves by algorithm name":
                group = p["algorithm"]

            if group not in groups_and_styles:
                groups_and_styles[group] = p
                groups_my[group]    = np.array(p["Y"])
                groups_mysqr[group] = (np.array(p["Y"]))**2
                groups_n[group]    = 1

                group_number = len(groups_order)

                groups_and_styles[group]["marker"]    = marker[group_number % len(marker)]
                groups_and_styles[group]["linestyle"] = linestyle[group_number % len(linestyle)]

                if analysisView.cbxDrawMarkers.checkState() != Qt.Checked:
                    groups_and_styles[group]["marker"] = None
                    groups_and_styles[group]["markevery"] = None

                groups_order.append(group)

            else:
                groups_and_styles[group]["aggregated"] += 1
                n = groups_n[group]
                min_len = min(len(p["Y"]), len(groups_my[group]))
                groups_my[group]    = np.array(p["Y"][0:min_len]) * 1.0/float(n+1)  + groups_my[group][0:min_len] * (n/float(n+1))
                groups_mysqr[group] = (np.array(p["Y"][0:min_len])**2) * 1.0/float(n + 1) + groups_mysqr[group][0:min_len] *  (n/float(n+1))
                groups_n[group]    += 1

        for group in groups_order:
            groups_sigmay = np.sqrt(groups_mysqr[group] - groups_my[group]**2)
            p = groups_and_styles[group]

            try:
                formatedLabel = analysisView.edtLegendFormatString.text().format(**p)
            except:
                showError("The legend contains unknown substitution variables")
                return

            x_filtered = p["X"][0:len(groups_sigmay)]

            if analysisView.cbxLegendLogScalingY.checkState() == Qt.Checked:
                ax.semilogy(x_filtered, groups_my[group], marker = p["marker"], markevery = p["markevery"], linestyle = p["linestyle"], label = formatedLabel, color = color[rasterizedPlots % len(color)])
                rasterizedPlots += 1

            if analysisView.cbxLegendLogScalingY.checkState() != Qt.Checked:
                ax.plot(x_filtered, groups_my[group], marker = p["marker"], markevery = p["markevery"], linestyle = p["linestyle"], label = formatedLabel, color = color[rasterizedPlots % len(color)])
                rasterizedPlots += 1

            if plot_error_bar:
                ax.fill_between(x_filtered, groups_my[group] - groups_sigmay, groups_my[group] + groups_sigmay, alpha=0.2)

            #ax.semilogy(p["X"], groups_my[group] - groups_sigmay, marker=p["marker"], markevery=p["markevery"], linestyle=p["linestyle"], label=p["label"] + f"(avg. {groups_n[group]} paths)")
            #ax.semilogy(p["X"], groups_my[group] + groups_sigmay, marker=p["marker"], markevery=p["markevery"], linestyle=p["linestyle"], label=p["label"] + f"(avg. {groups_n[group]} paths)")
    # ===================================================================================================================================
    ax.grid(True)
    legend = None

    if analysisView.cbxLegendFormat.currentText() == "no legend":
        pass

    elif analysisView.cbxLegendFormat.currentText() == "show horizontal" and export_legend_to_file==False:
        legend = ax.legend(loc='best', fontsize=fontSize, ncol=rasterizedPlots)

    elif analysisView.cbxLegendFormat.currentText() == "show vertical" and export_legend_to_file==False:
        legend = ax.legend(loc='best', fontsize=fontSize, ncol=1)

    elif analysisView.cbxLegendFormat.currentText() == "show horizontal" and export_legend_to_file==True:
        legend = ax.legend(loc='best', fontsize=fontSize, framealpha=1.0, frameon=True, ncol=rasterizedPlots)

    elif analysisView.cbxLegendFormat.currentText() == "show vertical" and export_legend_to_file==True:
        legend = ax.legend(loc='best', fontsize=fontSize, framealpha=1.0, frameon=True, ncol=1)

    # Export legend to file
    if export_legend_to_file and legend != None:
        fig = legend.figure
        # This guarantees the placement of a a legend so legend.get_window_extent() will work correctly
        fig.canvas.draw()

        bbox = legend.get_window_extent()
        extra_offset = -5
        expand = [-extra_offset, -extra_offset, extra_offset, extra_offset]

        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

        fileName, _ = QFileDialog.getSaveFileName(mainView, "Export legend to separate file",
                                                            "",
                                                            "Adobe Portable Document Format (*.pdf);;Portable Network Graphics (*.png)")
        if len(fileName) != 0:
            if not fileName.endswith(".pdf") and not fileName.endswith(".png"):
                fileName = fileName + ".pdf"

            fig.savefig(fileName, dpi="figure", bbox_inches=bbox)

    # Draw plot
    plt_cur.draw()


def onUpdateTableForAnalysis():
    w = analysisView.tblExperiments

    header = analysisView.tblExperiments.horizontalHeader()
    headerItems = []
    for i in range(len(header)):
        headerItems.append(analysisView.tblExperiments.horizontalHeaderItem(i).text())

    all_jobs = []

    sim_stats_res_lock.acquire()
    for job_id in simulations_by_rows:
        all_jobs.append(job_id)
    sim_stats_res_lock.release()

    while w.rowCount() < len(all_jobs):
        table_widget_items = []
        for k in range(len(headerItems)):
            table_widget_items.append(QtWidgets.QTableWidgetItem())
        w.insertRow(w.rowCount())
        r = w.rowCount() - 1

        for c, item in enumerate(table_widget_items):
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            w.setItem(r, c, item)

    while w.rowCount() > len(all_jobs):
        w.removeRow(w.rowCount() - 1)

    sim_stats_res_lock.acquire()

    for i in range(len(all_jobs)):
        job_id = all_jobs[i]
        sim_stats = simulations_stats[job_id]
        for j, header in enumerate(headerItems):
            if w.item(i, j) == None:
                w.setItem(i, j, QtWidgets.QTableWidgetItem(""))

            if header in sim_stats:
                w.item(i, j).setText(str(sim_stats[header]))
            else:
                w.item(i, j).setText("")

        # Fill cbxbAddExperimentColumn with options if it has not been filled
        if analysisView.cbxbAddExperimentColumn.count() == 0:
            for k in sim_stats.keys():
                if k == "args" or k == "H":
                    continue
                else:
                    analysisView.cbxbAddExperimentColumn.addItem(k)

    sim_stats_res_lock.release()


def customizeMatplotLib():
    matplotlib.rcParams['lines.linewidth'] = analysisView.sbxLineWidthSize.value()
    matplotlib.rcParams['lines.markersize'] = analysisView.sbxDefaultMarkerSizes.value()
    matplotlib.rcParams['font.size'] = analysisView.sbxFontSizes.value()

    # Ticks fonts
    # matplotlib.rcParams['xtick.labelsize'] = analysisView.sbxFontSizes.value()
    # matplotlib.rcParams['ytick.labelsize'] = analysisView.sbxFontSizes.value()

    # Turn off logging from font manager
    logging.getLogger('matplotlib.font_manager').disabled = True

    # The default font family is set with the font.family rcparam
    # matplotlib.rcParams['font.family'] = 'sans-serif'
    #  list of font styles to try to find in order:
    # matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Lucida Grande', 'Verdana']
    # matplotlib.font_manager.findSystemFonts(fontpaths = os.path.join(os.path.dirname(__file__), "./resources/fonts"), fontext='ttf')


def onExit():
    global earlyStopSimulations, simulations_stats

    # Register for early stopping all jobs
    sim_stats_res_lock.acquire()
    earlyStopSimulations = set(simulations_stats.keys())
    sim_stats_res_lock.release()

    # Wait for job finish
    simulationThreadPool.stop()

    # Quit application
    app.quit()


def onSelectAllExtraQuantitiesToTrack():
    to_check = (configView.cbxTrackObjectiveFunctionValueVal.checkState() ==  Qt.Checked)
    to_check = not to_check

    configView.cbxTrackObjectiveFunctionValueVal.setChecked(to_check)
    configView.cbxTrackObjectiveFunctionValueTrain.setChecked(to_check)
    configView.cbxTrackNormFullGradientTrain.setChecked(to_check)
    configView.cbxTrackNormFullGradientVal.setChecked(to_check)


def onSelectAllGpus():
    for i in range(configView.loComputeDevices.count()):
        w = configView.loComputeDevices.itemAt(i).widget()
        if w.text().find("gpu") >= 0:
            w.setCheckState(Qt.Checked)


def onCbxComputeDeviceChanged(index):
    curText = str(configView.cbxComputeDevice.currentText())
    if curText == "Several compute devices":
        configView.loComputeDevicesWidget.setVisible(True)
        configView.btnSelectAllGpus.setVisible(True)
    else:
        configView.loComputeDevicesWidget.setVisible(False)
        configView.btnSelectAllGpus.setVisible(False)


def onCurrentTabChanged(index):
    if index == 2:
        onUpdateTableForAnalysis()


def onCleanCurrentPlot():
    plt_cur = analysisView.customPlots[ analysisView.tabWidget.currentIndex() ]
    cleanPlot(plt_cur, analysisView.tabWidget.currentIndex())


def updateMemoryUsageStatus():
    import os, psutil

    current_device_index = configView.cbxComputeDevice.currentIndex()

    process = psutil.Process(os.getpid())
    mem = psutil.virtual_memory()

    cpu_acive_mbytes = (process.memory_info().rss + process.memory_info().vms) / (1024. ** 2)    # Size for the program with virtual allocated memory and resident memory
    cpu_all_gbytes = mem.total / (1024. ** 3)                                                    # All physical RAM size
    cpu_load_fraction = cpu_acive_mbytes / (cpu_all_gbytes * 1024.0) * 100.0                     # Percent of used CPU DRAM memory by process

    totalSwapBytes, usedSwapBytes, freeSwapBytes, percentSwapUsage, _, _ = psutil.swap_memory()

    cpuMem = " CPU {0:.2f} MB (RSS,VMS) / {1:.2f} GB (Total) [{2:.0f}%] / SYSTEM SWAP {3:.2f} MB/{4:.2f} GB [{5:.0f}%]".format(cpu_acive_mbytes, cpu_all_gbytes, cpu_load_fraction,
                                                                                                                         usedSwapBytes/(1024.0**2), totalSwapBytes/(1024.0**3), percentSwapUsage)


    gpuMem = []
    gpus_properties = gpu_utils.get_available_gpus()
    for i in range(len(gpus_properties)):
        device = "cuda:" + str(i)
        memory_gpu = torch.cuda.memory_stats(device)['reserved_bytes.all.current']
        acive_mbytes = memory_gpu / (1024. ** 2)
        all_gbytes   = gpus_properties[i].total_memory / (1024. ** 3)
        load_fraction = acive_mbytes / (all_gbytes * 1024.0) * 100.0           # Percent of used GPU DRAM memory by process
        gpuMem.append(" GPU{0} {1:.2f} MB/{2:.2f} GB [{3:.0f}%]".format(i, acive_mbytes, all_gbytes, load_fraction))

    # Selection all devices
    if current_device_index == configView.cbxComputeDevice.count() - 1:
        # device #0     -- cpu
        if configView.loComputeDevices.itemAt(0).widget().checkState() == Qt.Checked:
            cpuMem = "<b>" + cpuMem + "</b> "

        # device #1, #2 -- GPUs
        for i in range(len(gpus_properties)):
            if configView.loComputeDevices.itemAt(i + 1).widget().checkState() == Qt.Checked:
                gpuMem[i] = "<b>" + gpuMem[i] + "</b>"
    else:
        if current_device_index >= 1:
            gpuMem[current_device_index - 1] = "<b>" + gpuMem[current_device_index - 1] + "</b>"
        else:
            cpuMem = "<b>" + cpuMem + "</b> "

    # Demonstrate only need info
    if configView.cbxShowGpuInfo.checkState() == Qt.Checked:
        if len(gpuMem) == 0:
            mainView.gpuLabel.setText("No available GPUs")
        else:
            mainView.gpuLabel.setText(" " + " | ".join(gpuMem) + " ")
    else:
        mainView.gpuLabel.setText("")

    if configView.cbxShowCpuInfo.checkState() == Qt.Checked:
        mainView.cpuLabel.setText(cpuMem)
    else:
        mainView.cpuLabel.setText("")

    if configView.cbxShowNumberOfWorks.checkState() == Qt.Checked:
        max_workers = simulationThreadPool.workers()
        cur_execute_tasks = 0 
        queued_tasks = 0
        for i in range(max_workers):
            work_items_for_thread_i = len(simulationThreadPool.threads[i].cmds)
            if work_items_for_thread_i > 0:
                # One task is executing right now
                cur_execute_tasks += 1
                # The rest of the tasks are in the queue for a specific thread
                queued_tasks += (work_items_for_thread_i - 1)

        msg = f"Simulations - progress: {cur_execute_tasks}, max simultaneous: {max_workers}, in queue: {queued_tasks}"
        mainView.simulationJobsLabel.setText(msg)
    else:
        mainView.simulationJobsLabel.setText("")


def getDefaultMachineDescription(host, ip, port):
    """Obtain default machine description, before officially requesting it"""
    return {"host": host,
            "ip": ip,
            "port": port,
            "online": False,
            "devices": [],
            "used_devices": []
            }


def addMachineToMachineList():
    """Add machine to the machines list and fill the table"""
    address = machinesView.edtMachine.text()
    address = address.strip()
    address = address.split(",")
    for i in range(len(address)):
        if address[i].find(":") == -1:
            showError("Provide <hostname>:port")
            return

    for i in range(len(address)):
        address_parts = address[i].split(":")
        if len(address_parts) != 2:
            showError("Provide <hostname>:port")
            return

        host = address_parts[0] 
        port = int(address_parts[1])
        ip = ""

        try:
            ip = socket.gethostbyname(host)
        except socket.gaierror as err:
            showError(f"Cannot resolve hostname: {host}, error: {err}")
            return

        # Update datastructures
        descr = getDefaultMachineDescription(host, ip, port)
        machineDescriptions.append(descr)

        # Work with table
        w = machinesView.tblMachines
        it_name = QtWidgets.QTableWidgetItem("")
        it_ip  = QtWidgets.QTableWidgetItem("")
        it_port = QtWidgets.QTableWidgetItem("")
        it_gpus = QtWidgets.QTableWidgetItem("")
        it_use_cpu = QtWidgets.QTableWidgetItem("")
        it_use_gpu1 = QtWidgets.QTableWidgetItem("")
        it_use_gpu2 = QtWidgets.QTableWidgetItem("")
        it_use_gpu3 = QtWidgets.QTableWidgetItem("")
        it_use_gpu4 = QtWidgets.QTableWidgetItem("")

        it_online = QtWidgets.QTableWidgetItem("")

        w.insertRow(w.rowCount())
        r = w.rowCount() - 1
        for c, item in enumerate((it_name, it_ip, it_port, it_gpus, it_use_cpu,
                                  it_use_gpu1, it_use_gpu2, it_use_gpu3, it_use_gpu4, it_online)):
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            w.setItem(r, c, item)

        it_name.setText(str(descr["host"]))
        it_ip.setText(str(descr["ip"]))
        it_port.setText(str(descr["port"]))
        it_gpus.setText(str(len([d for d in descr["devices"] if d.find("cpu") == -1])))
        it_online.setData(QtCore.Qt.UserRole + 2000, descr["online"])
        # https://doc.qt.io/archives/qtjambi-4.5.2_01/com/trolltech/qt/core/Qt.ItemFlag.html
        for i, item in enumerate((it_use_cpu, it_use_gpu1, it_use_gpu2, it_use_gpu3, it_use_gpu4)):
            item.setCheckState(Qt.Unchecked)
            item.setFlags(QtCore.Qt.NoItemFlags)

        onUpdateStatusForExternalResources()


def removeSelectedMachines():
    """Remove some subset of machines"""

    selected_items = machinesView.tblMachines.selectedItems()
    rows = list(set([item.row() for item in selected_items]))
    rows.sort()
    rows.reverse()
    w = machinesView.tblMachines
    for r in rows:
        w.removeRow(r)
        del machineDescriptions[r]

    # print(rows)
    # print(machineDescriptions)


def removeAllRemoteMachines():
    """Remove all remote machines from GUI interface"""
    w = machinesView.tblMachines
    while w.rowCount() > 0:
        w.removeRow(w.rowCount() - 1)
    machineDescriptions = []


def onTblMachinesChanged(row, column):
    item = machinesView.tblMachines.item(row, column)
    currentState = item.checkState()

    columnName = machinesView.tblMachines.horizontalHeaderItem(column).text().lower()
    if columnName.find("cpu:") == -1 and columnName.find("gpu:") == -1:
        return

    devName = columnName.replace("use", "").strip()

    # Update device status
    used_devices = machineDescriptions[row]["used_devices"]
    if currentState == QtCore.Qt.Checked:
        if devName not in used_devices:
            used_devices.append(devName)
    else:
        if devName in used_devices:
            used_devices.remove(devName)
    machineDescriptions[row]["used_devices"] = used_devices


def spbParallelSimulationsValueChanged():
    """Callback for change value of spinbox"""
    if simulationThreadPool.workers() == configView.spbParallelSimulations.value():
        return

    is_any_task_in_process = False
    for i in range(simulationThreadPool.workers()):
        if len(simulationThreadPool.threads[i].cmds) > 0:
            is_any_task_in_process = True
            break

    if is_any_task_in_process == False:
        simulationThreadPool.adjust_num_workers(configView.spbParallelSimulations.value(), device_list=["cpu"])  
    else:
        dlg = QMessageBox()
        dlg.setWindowTitle("Information")
        dlg.setText(f"The simulation is in progress. Please adjust number of of parallel simulations after finish")
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setIcon(QMessageBox.Information)
        button = dlg.exec_()           

        configView.spbParallelSimulations.setValue(simulationThreadPool.workers())


def onTabWidgetInAnalysisChanged():
    """Callback for change current index of canvas in plots"""
    index = analysisView.tabWidget.currentIndex()

    # Save context for previous canvas
    # ==================================================================================================================
    prev_index = analysisView.prevPlotsNumber
    analysisView.customPlotsStates[prev_index] = \
        {"xaxe": analysisView.cbxXaxis.currentIndex(),
         "yaxe": analysisView.cbxYaxis.currentIndex(),

         "cbxLegendLogScalingY": analysisView.cbxLegendLogScalingY.checkState(),
         "cbxSyncTitleWithTabName": analysisView.cbxSyncTitleWithTabName.checkState(),
         "cbxDrawMarkers": analysisView.cbxDrawMarkers.checkState(),

         "sbxLineWidthSize": analysisView.sbxLineWidthSize.value(),
         "sbxDefaultMarkerSizes": analysisView.sbxDefaultMarkerSizes.value(),
         "sbxFontSizes": analysisView.sbxFontSizes.value(),

         "cbxAggregateCurves": analysisView.cbxAggregateCurves.currentIndex(),
         "cbxIncludeToLegend": analysisView.cbxIncludeToLegend.currentIndex(),
         "edtLegendFormatString": analysisView.edtLegendFormatString.text()
        }

    # Load context for current canvas
    # ==================================================================================================================
    if index in analysisView.customPlotsStates:
        s = analysisView.customPlotsStates[index]
        analysisView.cbxXaxis.setCurrentIndex(s["xaxe"])
        analysisView.cbxYaxis.setCurrentIndex(s["yaxe"])

        analysisView.cbxLegendLogScalingY.setCheckState(s["cbxLegendLogScalingY"])
        analysisView.cbxSyncTitleWithTabName.setCheckState(s["cbxSyncTitleWithTabName"])
        analysisView.cbxDrawMarkers.setCheckState(s["cbxDrawMarkers"])

        analysisView.sbxLineWidthSize.setValue(s["sbxLineWidthSize"])
        analysisView.sbxDefaultMarkerSizes.setValue(s["sbxDefaultMarkerSizes"])
        analysisView.sbxFontSizes.setValue(s["sbxFontSizes"])

        analysisView.cbxAggregateCurves.setCurrentIndex(s["cbxAggregateCurves"])
        analysisView.cbxIncludeToLegend.setCurrentIndex(s["cbxIncludeToLegend"])
        analysisView.edtLegendFormatString.setText(s["edtLegendFormatString"])

    # ==================================================================================================================
    analysisView.prevPlotsNumber = index


def tblExperimentsItemChanged(tblExperimentItem):
    r = tblExperimentItem.row()
    c = tblExperimentItem.column()
    new_job_id = tblExperimentItem.text()

    if new_job_id == "":
        return

    if c != 0:
        return

    global sim_stats_res_lock
    global earlyStopSimulations
    global simulations_stats
    global simulations_result
    global simulations_by_rows

    if sim_stats_res_lock.acquire(blocking=False) == False:
        return

    if r >= len(simulations_by_rows) or simulations_by_rows[r] == new_job_id or (new_job_id in simulations_stats) or (new_job_id in simulations_stats):
        sim_stats_res_lock.release()
        return
    else:
        old_job_id = simulations_by_rows[r]

        if simulations_stats[old_job_id]["finished"] == False:
            # For not finished jobs it's not allowable to change id for simplicity
            sim_stats_res_lock.release()
            return

        if old_job_id in earlyStopSimulations:
            earlyStopSimulations.remove(old_job_id)
            earlyStopSimulations.add(new_job_id)

        if old_job_id in simulations_stats:
            simulations_stats[new_job_id] = simulations_stats[old_job_id]
            simulations_stats[new_job_id]["run_id"] = new_job_id

            del simulations_stats[old_job_id]

        if old_job_id in simulations_result:
            simulations_result[new_job_id] = simulations_result[old_job_id]
            simulations_result[new_job_id]["run_id"] = new_job_id
            simulations_result[new_job_id]["args"].run_id = new_job_id

            del simulations_result[old_job_id]

        simulations_by_rows[r] = new_job_id

        sim_stats_res_lock.release()


# ======================================================================================================================
# Several subclass with customized application's windows
# For customize logic - you can override values from your UI file within your code, or if possible set in Qt Creator.
# ======================================================================================================================
class MyMainView(QtWidgets.QMainWindow, MainView.Ui_MainView):
    def __init__(self, *args, **kwargs):
        super(MyMainView, self).__init__(*args, **kwargs)
        self.setupUi(self)

    def closeEvent(self, event):
        global earlyStopSimulations, simulations_stats
        not_saved_experiments = []

        sim_stats_res_lock.acquire()
        for row in range(len(simulations_by_rows)):
            job_id = simulations_by_rows[row]
            if "source" not in simulations_stats[job_id] or len(simulations_stats[job_id]["source"]) == 0:
                not_saved_experiments.append(row)
        sim_stats_res_lock.release()

        if not_saved_experiments:
            dlg = QMessageBox()
            dlg.setWindowTitle("Do you want to exit without saving all experiments?")
            dlg.setText(f"Are you sure you want to exit? Experiments in rows: {not_saved_experiments} are not saved")
            dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            dlg.setIcon(QMessageBox.Question)
            button = dlg.exec_()
            if button == QMessageBox.No:
                event.ignore()
                return

        # ==============================================================================================================
        print("Register for early stopping all jobs")
        sim_stats_res_lock.acquire()
        earlyStopSimulations = set(simulations_stats.keys())
        sim_stats_res_lock.release()
        print("Wait for job to finish")
        simulationThreadPool.stop()
        # ==============================================================================================================
        event.accept()


class MyConfigWidget(QtWidgets.QWidget, ConfigWidget.Ui_ConfigWidget):
    def __init__(self, *args, **kwargs):
        super(MyConfigWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)


class MySimulationWidget(QtWidgets.QWidget, SimulationWidget.Ui_SimulationWidget):
    def __init__(self, *args, **kwargs):
        super(MySimulationWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)


class MyLogView(QtWidgets.QMainWindow, LogWindow.Ui_LogWindow):
    def __init__(self, *args, **kwargs):
        super(MyLogView, self).__init__(*args, **kwargs)
        self.setupUi(self)


class MyMultiMachineSelector(QtWidgets.QMainWindow, MultiMachineSelector.Ui_MultiMachineSelector):
    def __init__(self, *args, **kwargs):
        super(MyMultiMachineSelector, self).__init__(*args, **kwargs)
        self.setupUi(self)


class MyAnalysisWidget(QtWidgets.QWidget, AnalysisWidget.Ui_AnalysisWidget):
    def __init__(self, *args, **kwargs):
        super(MyAnalysisWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)


if __name__ == "__main__":
    # ======================================================================================================================
    # ENTRY POINT FOR GUI
    # ======================================================================================================================
    # QApplication instance per application
    # ======================================================================================================================
    app = QtWidgets.QApplication(sys.argv)
    app.lastWindowClosed.connect(onExit)

    # ======================================================================================================================
    # Create main windows
    # ======================================================================================================================
    mainView = MyMainView()
    configView = MyConfigWidget()
    simulationView = MySimulationWidget()
    logView = MyLogView()
    analysisView = MyAnalysisWidget()
    machinesView = MyMultiMachineSelector()

    # ======================================================================================================================
    # Add styles and font selection
    # ======================================================================================================================
    stylesStrings = []
    styles = QtWidgets.QStyleFactory.keys()
    squares = []

    for s in styles:
        actionSelectStyle = QtWidgets.QAction(mainView)
        actionSelectStyle.setText(s)
        mainView.menuStyle.addAction(actionSelectStyle)
        actionSelectStyle.triggered.connect(lambda checked, selectedStyle=s: app.setStyle(selectedStyle))
        stylesStrings.append(s)

    if 'Fusion' in stylesStrings:
        app.setStyle('Fusion')


    def makeFixedFontSize(fontSize):
        font = app.font()
        font.setPointSize(fontSize)
        app.setFont(font)


    def customFontSelection():
        font, ok = QFontDialog.getFont()
        if ok:
            app.setFont(font)


    mainView.action_font_4.triggered.connect(lambda: makeFixedFontSize(4))
    mainView.action_font_6.triggered.connect(lambda: makeFixedFontSize(6))
    mainView.action_font_8.triggered.connect(lambda: makeFixedFontSize(8))
    mainView.action_font_10.triggered.connect(lambda: makeFixedFontSize(10))
    mainView.action_font_12.triggered.connect(lambda: makeFixedFontSize(12))
    mainView.action_font_select_Custom.triggered.connect(customFontSelection)


    # ======================================================================================================================
    # Configure default view for Artificial dataset generation
    # ======================================================================================================================
    showArtificialDatasetParameters(False)

    # ======================================================================================================================
    # Setup extra plot widgets
    # ======================================================================================================================
    analysisView.customPlotsStates = {}
    analysisView.prevPlotsNumber = 0

    analysisView.customPlots = []

    layouts_to_append_plots = [analysisView.loCustomPlot_1, analysisView.loCustomPlot_2, analysisView.loCustomPlot_3,
                               analysisView.loCustomPlot_4, analysisView.loCustomPlot_5, analysisView.loCustomPlot_6,
                               analysisView.loCustomPlot_7]

    for i in range(len(layouts_to_append_plots)):
        customPlot = MplCanvas(analysisView, width=5, height=4, dpi=100)
        analysisView.customPlots.append(customPlot)
        layouts_to_append_plots[i].addWidget(customPlot)
        layouts_to_append_plots[i].addWidget(customPlot.toolbar)

    customizeMatplotLib()

    for i, plot in enumerate(analysisView.customPlots):
        cleanPlot(plot, i)

    analysisView.btnUpdateTable.clicked.connect(onUpdateTableForAnalysis)
    analysisView.btnPlot.clicked.connect(lambda: onPlotResults(export_legend_to_file=False))
    analysisView.btnPlotAndExportLegend.clicked.connect(lambda: onPlotResults(export_legend_to_file=True))

    analysisView.tabWidget.currentChanged.connect(onTabWidgetInAnalysisChanged)
    analysisView.btnMoveSimulationUp.clicked.connect(moveSimulationUp)
    analysisView.btnMoveSimulationDown.clicked.connect(moveSimulationDown)
    analysisView.btnIncludeToLegend.clicked.connect(includeToLegend)

    # Setup relations between widgets - append separate widgets into need places of a common skeleton
    mainView.loConfigTab.addWidget(configView)
    mainView.loSimulation.addWidget(simulationView)
    mainView.loAnalysis.addWidget(analysisView)

    # Create all need signal/slots
    logView.actionClean.triggered.connect(lambda: logView.txtMain.clear())
    logView.actionSysInfo.triggered.connect(onLogSystemInformation)
    logView.actionTorchInfo.triggered.connect(lambda: uiLogInfo(get_pretty_env_info()))
    logView.actionGarbageCollector.triggered.connect(onCleanupMemory)
    logView.actionMemInfo.triggered.connect(onMemoryInfo)

    # Command line generation
    logView.actionCmdLineGeneration.triggered.connect(
        lambda: onCmdLineGenerationForFinishedExperiments(withNewLines=True)
    )

    logView.actionCmdLineGeneration.triggered.connect(
        lambda: onCmdLineGenerationForCurrentExperiment(withNewLines=True)
    )

    logView.actionCmdLineSingleGeneration.triggered.connect(
        lambda: onCmdLineGenerationForFinishedExperiments(withNewLines=False)
    )

    logView.actionCmdLineSingleGeneration.triggered.connect(
        lambda: onCmdLineGenerationForCurrentExperiment(withNewLines=False)
    )

    logView.actionExperimentInfoGeneration.triggered.connect(onExperimentInfoGeneration)

    logView.actionExitLogWindow.triggered.connect(onExitLogWindow)

    # Signals/slots relative to add machines
    machinesView.btnAddMachine.clicked.connect(addMachineToMachineList)
    machinesView.edtMachine.returnPressed.connect(addMachineToMachineList)

    machinesView.btnRemoveSelected.clicked.connect(removeSelectedMachines)
    machinesView.btnCleanAll.clicked.connect(removeAllRemoteMachines)
    machinesView.btnExit.clicked.connect(onExitMachineSelectorWindow)

    # ======================================================================================================================
    header = machinesView.tblMachines.horizontalHeader()
    for i in range(len(header) - 1):
        header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)

    header.setSectionResizeMode(len(header) - 1, QtWidgets.QHeaderView.Stretch)

    # Setup delegate painter for last column as icon
    delegate = StatusDelegate(machinesView.tblMachines)
    machinesView.tblMachines.setItemDelegateForColumn(machinesView.tblMachines.columnCount() - 1, delegate)

    machinesView.tblMachines.cellChanged.connect(onTblMachinesChanged)
    machinesView.btnRefreshMachineStatus.clicked.connect(onUpdateStatusForExternalResources)
    # ======================================================================================================================

    mainView.actionAbout.triggered.connect(onShowAboutDialog)

    mainView.actionLogWindow.triggered.connect(lambda: logView.show() or logView.raise_())
    mainView.actionViewMachines.triggered.connect(lambda: machinesView.show() or machinesView.raise_())

    mainView.actionExit.triggered.connect(onExit)
    mainView.actionNextTab.triggered.connect(onMoveToNextTabInMain)
    mainView.actionPrevTab.triggered.connect(onMoveToPrevTabInMain)

    mainView.actionLoad.triggered.connect(onLoadExperimentResults)
    mainView.actionSave.triggered.connect(lambda: onSaveExperimentResults(selected=False))
    mainView.actionactionSaveSelected.triggered.connect(lambda: onSaveExperimentResults(selected=True))


    simulationView.btnStart.clicked.connect(onStartSimulation)
    simulationView.btnCancel.clicked.connect(onCancelSimulation)
    simulationView.btnCancelAll.clicked.connect(onCancelAllSimulations)
    simulationView.btnClean.clicked.connect(onCleanCheckpointAndLogs)
    simulationView.btnStart.setShortcut("F5")

    simulationView.btnRefreshInformationAboutExp.clicked.connect(onLoadSummaryTable)
    simulationView.tblExperiments.itemSelectionChanged.connect(onLoadSummaryTable)
    simulationView.btnRemoveExperiments.clicked.connect(onRemoveAllExperiments)
    simulationView.btnRemoveSelectedExperiments.clicked.connect(onRemoveSelectedExperiment)
    simulationView.edtFieldsThatContains.textChanged.connect(onLoadSummaryTable)
    simulationView.tblExperiments.itemSelectionChanged.connect(onChangeSelectedExperimentsInSimView)
    analysisView.tblExperiments.itemSelectionChanged.connect(onChangeSelectedExperimentsInAnalysisView)

    analysisView.btnCleanExperimentsColumns.clicked.connect(onBtnCleanExperimentsColumns)
    analysisView.btnAddExperimentsColumn.clicked.connect(onBtnAddExperimentsColumn)

    # Add to configuration view all optimization algorithms
    for a in algorithms.getAlgorithmsList():
        configView.cbxOptAlgo.addItem(a.upper())

    configView.cbxClientCompressor.currentIndexChanged.connect(
        lambda index: configView.edtClientCompressorFormat.setText(compressorFormats[index])
    )

    # Initialize compressor format
    configView.edtClientCompressorFormat.setText(compressorFormats[configView.cbxClientCompressor.currentIndex()])

    configView.cbxDataset.currentIndexChanged.connect(onSelectDataset)
    configView.btnTrackAll.clicked.connect(onSelectAllExtraQuantitiesToTrack)
    configView.cbxComputeDevice.currentIndexChanged.connect(onCbxComputeDeviceChanged)
    configView.btnSelectAllGpus.clicked.connect(onSelectAllGpus)

    mainView.tabMainView.currentChanged.connect(onCurrentTabChanged)
    analysisView.btnCleanPlot.clicked.connect(onCleanCurrentPlot)
    configView.btnLaunchSimulation.clicked.connect(onStartSimulation)
    configView.btnLaunchSimulation.setShortcut("F5")

    configView.cbxOptAlgo.currentIndexChanged.connect(cbxOptAlgorithmSelection)

    configView.btnGenerateInitSeed.clicked.connect(
        lambda: configView.edtRandomInitSeed.setText(str(gui_rnd.randint(1, 10**9)))
    )

    configView.btnGenerateRunSeed.clicked.connect(
        lambda: configView.edtRandomRunSeed.setText(str(gui_rnd.randint(1, 10**9)))
    )

    # Change sampling type handler
    configView.cbxClientSamlingType.currentIndexChanged.connect(cbxClientSamplingTypeSelection)
    configView.lblPoissonSampling.setVisible(False)
    configView.edtPoissonSampling.setVisible(False)

    # Adjust number of workers coincident with UI
    simulationThreadPool.adjust_num_workers(configView.spbParallelSimulations.value(), device_list=["cpu"])
    configView.spbParallelSimulations.valueChanged.connect(spbParallelSimulationsValueChanged)

    timer = QTimer()
    timer.setInterval(1500)
    timer.timeout.connect(onTimerEvent)
    timer.start()

    resTimer = QTimer()
    resTimer.setInterval(10000)
    resTimer.timeout.connect(onUpdateStatusForExternalResources)
    resTimer.start()

    # Register need callbacks for obtaining information during simulation
    utils.execution_context.is_simulation_need_earlystop_fn = isSimulationNeedEarlyStop
    utils.execution_context.simulation_start_fn = simulationStart
    utils.execution_context.simulation_finish_fn = simulationFinish
    utils.execution_context.simulation_progress_steps_fn = simulationProgressSteps

    # Fill widget with runtime information
    device_list = []
    (system, node, release, version, machine, processor) = platform.uname()

    device_list.append(f"cpu - {processor}")
    gpus_properties = gpu_utils.get_available_gpus()
    for i in range(len(gpus_properties)):
        device_list.append(f"gpu:{i} - {gpus_properties[i].name} / {(gpus_properties[i].total_memory / (1024.0 ** 3)):.2f} GBytes")
    configView.cbxComputeDevice.addItems(device_list + ["Several compute devices"])

    if len(device_list) >= 2:
        configView.cbxComputeDevice.setCurrentIndex(1)

    # Create a a possible selection for selected external devices
    for dev in device_list:
        configView.loComputeDevices.addWidget(QtWidgets.QCheckBox(dev))
    configView.loComputeDevices.addWidget(QtWidgets.QCheckBox("external network resources"))

    # Update view based on current dataset selection
    onSelectDataset(configView.cbxDataset.currentIndex())

    # Setup delegate painter for last column to draw progress bar
    delegate = ProgressDelegate(simulationView.tblExperiments)
    simulationView.tblExperiments.setItemDelegateForColumn(simulationView.tblExperiments.columnCount() - 1, delegate)

    # Setup callback for change id of the JOB in UI
    simulationView.tblExperiments.itemChanged.connect(tblExperimentsItemChanged)

    # Setup resizing of a table columns
    header = simulationView.tblExperiments.horizontalHeader()
    header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
    header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
    header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

    header = simulationView.tblSummary.horizontalHeader()
    header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
    header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

    header = analysisView.tblExperiments.horizontalHeader()
    for i in range(len(header) - 1):
        header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
    header.setSectionResizeMode(len(header) - 1, QtWidgets.QHeaderView.Stretch)
    header.sectionClicked.connect(onHeaderAnalysisExperimentClicked)

    # Labels for memory CPU/GPU usage
    mainView.cpuLabel = QtWidgets.QLabel()
    mainView.gpuLabel = QtWidgets.QLabel()
    mainView.simulationJobsLabel = QtWidgets.QLabel()

    mainView.cpuLabel.setStyleSheet("QLabel {background-color: rgb(217,217,255)}")
    mainView.gpuLabel.setStyleSheet("QLabel  {background-color: rgb(64,255,64)}")
    mainView.simulationJobsLabel.setStyleSheet("QLabel  {background-color: rgb(64,255,64)}")

    mainView.cpuLabel.setTextFormat(Qt.TextFormat.RichText)
    mainView.gpuLabel.setTextFormat(Qt.TextFormat.RichText)
    mainView.simulationJobsLabel.setTextFormat(Qt.TextFormat.RichText)

    mainView.memToolBar.addWidget(mainView.cpuLabel)
    mainView.memToolBar.addWidget(mainView.gpuLabel)
    mainView.memToolBar.addWidget(mainView.simulationJobsLabel)

    updateMemoryUsageStatus()

    # ======================================================================================================================
    # Run main event loop
    show_maximized = True
    mainView.setWindowTitle("FL_PyTorch")

    if show_maximized:
        mainView.showMaximized()
    else:
        # Show window. Windows are hidden by default.
        mainView.show()

    # ======================================================================================================================
    # Start the event loop
    # ======================================================================================================================
    sys.exit(app.exec_())
