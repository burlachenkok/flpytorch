#!/usr/bin/env python3

import threading
import time


class Buffer:
    """Thread safe ring - buffer of commands"""

    def __init__(self, max_capacity=100):
        self.max_capacity = max_capacity
        self.items = [None] * self.max_capacity  # List with reserved memory to eliminate problems with memory resizing
        self.front_index = 0  # Read position for popFront
        self.back_index = 0  # Write position for pushBack
        self.length = 0  # Length of buffer in elements
        self.lock = threading.Lock()  # Be default lock is not acquired
        self.item_is_ready = threading.Semaphore(value=0)  # Semaphore to signal when there are items in the buffer

    def __len__(self):
        """Total number of items in the thread-safe container."""
        self.lock.acquire()
        L = self.length
        self.lock.release()

        return L

    def isFull(self):
        """Check that buffer is full"""
        return len(self) == self.max_capacity

    def isEmpty(self):
        """Predict which allow to check is container empty."""
        return len(self) == 0

    def pushBack(self, item):
        """
      Push item into the back of a container, with blocking.

      Args:
          item (object): item which is inserted into container
      """
        while True:
            self.lock.acquire()
            if self.length == self.max_capacity:
                self.lock.release()
                time.sleep(0.0001)
                continue

            else:
                self.items[self.back_index] = item
                self.back_index = (self.back_index + 1) % self.max_capacity
                self.length += 1
                self.lock.release()
                self.item_is_ready.release()
                break

        return self

    def waitForItem(self):
        """Wait for item available for popping from container via popFront(), blocking."""
        self.item_is_ready.acquire()
        return self

    def popFront(self):
        """
          Get item from the front of a container, no blocking.

          Method does not perform checking is any element in the container is available.
          Please use waitForItem() or len() if you're not sure.

          Returns:
          object: item from a container
        """
        self.lock.acquire()
        return_item = self.items[self.front_index]
        self.front_index = (self.front_index + 1) % self.max_capacity
        self.length -= 1
        self.lock.release()
        return return_item

    def front(self):
        """Get item from the front of a container, not blocking."""
        self.lock.acquire()
        return_item = self.items[self.front_index]
        self.lock.release()
        return return_item

    def get(self, index):
        """
      Get item from the front of a container, not blocking

      Method does not perform checking is index within need range accessible range.

      Returns:
      object: item from a container
      """
        self.lock.acquire()
        index = index + self.front_index
        index = index % self.max_capacity
        return_item = self.items[index]
        self.lock.release()
        return return_item

    def __getitem__(self, index):
        return self.get(index)


# ======================================================================================================================
# Unittests for launch please use: "pytest -v buffer.py" 
# https://docs.pytest.org/en/stable/getting-started.html

def test_cmd_buffer_push_pop():
    b = Buffer()
    b.pushBack(10).pushBack(20)
    assert len(b) == 2
    assert b.popFront() == 10
    assert b.popFront() == 20
    assert len(b) == 0
    assert b.isEmpty()


def test_cmd_buffer_waiting():
    b = Buffer()
    assert b.isEmpty()

    b.pushBack(10)
    b.pushBack(20)
    b.pushBack(30)
    assert not b.isEmpty()
    b.waitForItem()
    assert b.front() == 10
    assert b.front() == 10
    assert b.popFront() == 10
    assert b.front() == 20
    assert b.popFront() == 20


def test_cmd_buffer_indexing():
    b = Buffer()
    b.pushBack(10).pushBack(20).pushBack(30)
    assert b[0] == 10
    assert b[1] == 20
    assert b[2] == 30

    b = Buffer()
    b.pushBack(50).pushBack(10).pushBack(20).pushBack(30)
    assert len(b) == 4
    assert 50 == b.popFront()
    assert b[0] == 10
    assert b[1] == 20
    assert b[2] == 30

    c = Buffer(max_capacity=3)
    c.pushBack(50).pushBack(10).pushBack(20)
    c.popFront()
    c.popFront()
    c.pushBack(21).pushBack(22)

    assert c[0] == 20
    assert c[1] == 21
    assert c[2] == 22
    assert len(c) == 3


def test_buffer_waiting():
    class TestThread(threading.Thread):
        def __init__(self, buffer):
            threading.Thread.__init__(self)
            self.buffer = buffer

        def run(self):
            time.sleep(1.0)
            out.pushBack("Action-2")
            self.buffer.popFront()

    b = Buffer(3)
    out = Buffer(3)
    b.pushBack(1)
    b.pushBack(2)
    assert not b.isFull()
    b.pushBack(3)
    assert b.isFull()
    th = TestThread(b)
    th.start()
    out.pushBack("Action-1")
    b.pushBack(4)
    out.pushBack("Action-3")
    assert out[0] == "Action-1"
    assert out[1] == "Action-2"
    assert out[2] == "Action-3"
# ======================================================================================================================
