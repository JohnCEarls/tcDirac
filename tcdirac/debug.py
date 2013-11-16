DEBUG = True
debug_dir='/scratch/sgeadmin/debug'
import time
import os
import os.path
import sys

import pickle
import logging
import logging.handlers
import SocketServer
import struct
import socket
from multiprocessing import Process, Event

import static

class TimeTracker:
    def __init__(self):
        self._wait_tick = time.time()
        self._work_tick = time.time()
        self._waiting = 0.0
        self._working = 0.0

    def start_work(self):
        self._work_tick= time.time()

    def end_work(self):
        self._working += time.time() - self._work_tick
        self._work_tick = time.time()


    def start_wait(self):
        self._wait_tick= time.time()

    def end_wait(self):
        self._waiting += time.time() - self._wait_tick
        self._wait_tick = time.time()

    def print_stats(self):
        print
        print "Waiting Time:", self._waiting
        print "Working Time:", self._working
        print "working/waiting", self._working/self._waiting
        print
"""
Below is lightly modified from http://docs.python.org/2/howto/logging-cookbook.html#logging-cookbook
"""


class LogRecordStreamHandler(SocketServer.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)

class LogRecordSocketReceiver(SocketServer.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver suitable for testing.
    """
    allow_reuse_address = 1
    def __init__(self, host='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT, handler=LogRecordStreamHandler):
        SocketServer.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()], [], [],  self.timeout)
            abort = self.abort
            if rd:
                self.handle_request()

def startLogger():
    try:
        LOG_FILENAME = "/scratch/sgeadmin/logs/tcdirac.log"
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(  format=log_format)
        handler = logging.handlers.RotatingFileHandler( LOG_FILENAME, maxBytes=2000000, backupCount=5)
        handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger('').addHandler(handler)
        tcpserver = LogRecordSocketReceiver()
        print('About to start TCP server...')
        tcpserver.serve_until_stopped()
    except:
        print
        pass

def initLogging(server='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT, server_level=static.logging_server_level, sys_out_level=static.logging_stdout_level):
    rootLogger = logging.getLogger('')
    rootLogger.setLevel(logging.DEBUG)
    socketHandler = logging.handlers.SocketHandler(server, port)
    socketHandler.setLevel(server_level)
    rootLogger.addHandler(socketHandler)
    if sys_out_level is not None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(sys_out_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        rootLogger.addHandler(ch)



if __name__ == "__main__":
        initLogging()
        logging.error("test")
        logger = logging.getLogger("test1-new")
        logger.error("test")

