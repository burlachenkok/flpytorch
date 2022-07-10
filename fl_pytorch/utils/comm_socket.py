#!/usr/bin/env python3

import socket


class CommSocket:
    def __init__(self, sock=None):
        """Ctor for socket communication socket object"""
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock
        # Turn off timeout
        self.sock.settimeout(None)

    def rawSend(self, msg):
        """
        Send a raw bytes over a network in form of <LEN:><BYTES>,
        where <LEN:> is encoded in text format and it total length of the message.

        Args:
            msg(bytes): A bytes like object

        Returns:
            None.
        """

        sent = self.sock.send(bytes(f"{len(msg)}:", "utf-8"))
        if sent == 0:
            raise RuntimeError("socket connection broken")
       
        msgLen = len(msg)
        totalsent = 0
        while totalsent < msgLen:
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            totalsent = totalsent + sent

    def rawSendString(self, msg):
        """Send a text string (encoded in UTF-8) over a network.

        Args:
            msg(str): text string that will be send over the network.

        Returns:
            None.
        """
        self.rawSend(bytes(msg, 'utf-8'))

    def rawRecvString(self):
        """Wait and receive string object
        Returns:
            Obtained string from the network connection.
        """
        return self.rawRecv().decode('utf-8')

    def rawRecv(self):
        """Wait and receive bytes-like object
        Returns:
            Bytes object.
        """
        prefix = []
        while True:
            chunk = self.sock.recv(1)
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            elif chunk == b':':
                break
            else:
                prefix.append(chunk.decode())

        MSGLEN = int(''.join(prefix))
        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            chunk = self.sock.recv(min(MSGLEN - bytes_recd, 2048))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b''.join(chunks)
        # ==============================================================================================================
