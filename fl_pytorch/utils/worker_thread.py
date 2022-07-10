#!/usr/bin/env python3

# Import PyTorch root package import torch
import torch

import threading

from . import buffer


class WorkerThread(threading.Thread):
    """Worker thread. It's goal execute deferred functions."""

    def __init__(self):
        threading.Thread.__init__(self)

        self.cmds = buffer.Buffer()

        self.completion_event_lock_event = threading.Lock()  # Be default lock is not acquired
        self.completion_event_lock_event.acquire()           # Acquire lock

    def defered_call(self, f, args):
        """
        Deferred execution of function, not blocking.

        Args:
            f (function): The deferred function to execute
            args (tuple): Arguments for function f which are needed to be passed

        Returns:
          None
        """
        function_and_args = (f, args)
        self.cmds.pushBack(function_and_args)

    def run(self):
        report_about_completion = False

        while True:
            if report_about_completion:
                if len(self.cmds) == 0:
                    report_about_completion = False
                    self.completion_event_lock_event.release()

            self.cmds.waitForItem()   # Wait for item in a work queue, blocking
            item = self.cmds.front()  # Get item without block 

            if type(item) == str:
                if item == "_STOP_":
                    return                          # We have obtained command to stop execution
                elif item == "_S_COMPLETE_":
                    report_about_completion = True  # Somebody waits for reporting once for event
                                                    # for report about finishing execution
            else:
                f, args = item  

                # Execute function possibly within own cuda stream context which will allow to
                # submit work into NVIDIA GPU without waiting previous works for another streams
                # IMPORTANT: right now retValue is ignored
                if hasattr(self, "worker_stream"):
                    with torch.cuda.stream(self.worker_stream):
                        retValue = f(self, *args)
                    # Force waiting for finishing write-back operations in GPU memory for that thread
                    self.worker_stream.synchronize()
                else:
                    retValue = f(self, *args)

            self.cmds.popFront()   # Get rid of from item in the queue

    def synchronize(self):
        """ Wait until thread process all queued tasks before that moment in CPU."""

        if len(self.cmds) == 0:
            return
        self.cmds.pushBack("_S_COMPLETE_")
        self.completion_event_lock_event.acquire()

        if hasattr(self, "worker_stream"):
            self.worker_stream.synchronize()

    def stop(self):
        """ Request thread after processing all queued tasks before that moment complete it's work."""
        self.cmds.pushBack("_STOP_")

    def stopAndJoin(self):
        """ Request thread after processing all queued tasks before that moment complete it's work."""
        self.cmds.pushBack("_STOP_")

        self.join()


# Unittests for launch please use: "pytest -v worker_thread.py"
# https://docs.pytest.org/en/stable/getting-started.html
def test_worker_thread():
    th1 = WorkerThread()
    z = 0

    def testf(thread, x, y): 
        nonlocal z
        z = x/y
    th1.start()
    th1.defered_call(testf, (6, 2))
    th1.synchronize()
    assert z == 3

    th1.defered_call(testf, (8, 2))
    th1.stopAndJoin()
    assert z == 4
# ======================================================================================================================
