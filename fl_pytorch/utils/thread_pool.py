#!/usr/bin/env python3

import time

# Import PyTorch root package import torch
import torch

from . import worker_thread
from . import gpu_utils


class ThreadPool:
    """Thread pool. Collectively execute assigned work."""

    def __init__(self, number_of_workers=0):
        """
        Constructor.

        Args:
            number_of_workers (int): number of working threads in a thread pool
        """
        self.number_of_workers = 0
        self.next_worker = 0
        self.threads = []
        self.adjust_num_workers(number_of_workers)

    def get_free_worker_index(self):
        """Get index of worker which currently do nothing or -1 if there are no such worker
        Returns:
            Integer index of free worker within a thread pool
        """
        for i in range(len(self.threads)):
            if len(self.threads[i].cmds) == 0:
                return i
        return -1

    def workers(self):
        """Get current number of workers within a threadpool
        Returns:
            number of workers within a thread pool
        """
        return self.number_of_workers

    def adjust_num_workers(self, w=0, own_cuda_stream=False, device_list=["cuda"]):
        """
        Adjust number of workers in a thread pool.
        Complete work for all existing workers and finish executing of threads and crete new thread pool.

        Args:
            number_of_workers (int): number of working threads in a thread pool
        """
        self.number_of_workers = w
        self.next_worker = 0
        self.stop()

        for i in range(w):
            # Dispatch GPU devices across workers uniformly
            # ==========================================================================================================
            device = device_list[i % len(device_list)]
            is_target_gpu = gpu_utils.is_target_dev_gpu(device)

            th = worker_thread.WorkerThread()

            if own_cuda_stream and is_target_gpu:
                th.worker_stream = torch.cuda.Stream(device)
            if not own_cuda_stream and is_target_gpu:
                th.worker_stream = torch.cuda.default_stream(device)

            th.own_cuda_stream = own_cuda_stream  # Flag that thread has it's own CUDA stream
            th.is_target_gpu = is_target_gpu      # Flag that target device is GPU
            th.device = gpu_utils.get_target_device_str(device)  # Get target device string
            # ==========================================================================================================
            th.start()
            self.threads.append(th)

    def next_dispatched_thread(self):
        """ Get reference to thread which will process next dispatch call. """
        th = self.threads[self.next_worker]
        return th

    def dispatch(self, function, args, worker_index=-1):
        """
        Dispatch  execution of function for one of the workers.

        Args:
            function(function): function to execute.
                                Obtains in first argument reference to the thread and in second arguments from the list.
            args(tuple): function arguments which will be pass for function with the reference to worker thread
            worker_index(ind): -1 use auto-dispatch, worker_index>=0 will lead to use specific worker thread

        Returns:
            True if dispatching happens fine.
            False if there are no threads in a thread pool or all threads have already complete their work.
        """

        if len(self.threads) == 0:
            return False

        if worker_index < 0:
            th = self.threads[self.next_worker]
            th.defered_call(function, args)
            self.next_worker = (self.next_worker + 1) % self.number_of_workers
        else:
            th = self.threads[worker_index]
            th.defered_call(function, args)

        return True

    def synchronize(self):
        """Synchronize via waiting for complete work execution for threads in a thread pool."""
        for th in self.threads:
            th.synchronize()

    def stop(self):
        """Completely finalize execution of all threads in a thread pool."""
        for th in self.threads:
            th.stop()

        for th in self.threads:
            th.join()
        self.threads = []


def test_thread_pool_no_work():
    p1 = ThreadPool(10)
    assert p1.workers() == 10
    p1.adjust_num_workers(3)
    assert p1.workers() == 3
    assert len(p1.threads) == 3
    p1.stop()


def test_thread_pool_with_work():
    def wait(th, seconds):
        time.sleep(seconds)

    p1 = ThreadPool(2)
    for i in range(10):
        assert p1.dispatch(wait, (0.100,))
    p1.stop()
    p1.synchronize()
    assert not p1.dispatch(wait, (0.100,))
# ======================================================================================================================
