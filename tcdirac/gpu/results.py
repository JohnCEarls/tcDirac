
import sys

import inspect, os, os.path
if os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) == '/home/sgeadmin/hdproject/tcDirac/tcdirac/gpu':
    #if we are running this from dev dir, need to add tcdirac to the path
    sys.path.append('/home/sgeadmin/hdproject/tcDirac')

import errno
import time
import logging
import itertools
from tempfile import TemporaryFile
import hashlib
import random
import cPickle

from multiprocessing import Process, Queue, Lock, Value, Event, Array
from Queue import Empty

import boto
from boto.s3.key import Key
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message

import ctypes
import numpy as np
import scipy.misc
import pandas

import pycuda.driver as cuda

import tcdirac.dtypes as dtypes
from tcdirac.gpu import processes



class Packer(Process):
    def __init__(self, name,p_type, in_q, out_q, smem,events, out_dir, dr_timeout=10):
        Process.__init__(self, name=name)
        self.in_q = in_q
        self.out_q = out_q
        self.shared_mem = smem
        self.events = events
        self.out_dir = out_dir
        self._dr_timeout = dr_timeout
        self.p_type = p_type


    def run(self):
        self.events['add_data'].set()
        wait_ctr = 0
        while True:
            self.events['data_ready'].wait(self._dr_timeout)
            if self.events['data_ready'].is_set():
                mess = {}
                try:
                    file_id = self.in_q.get(True, 3)
                    mess['file_id'] = file_id
                    data = self.get_data()
                    #Note: this is not guaranteed to be unique, only part of array used
                    #      do not want to wait for large matrix to be hashed
                    f_hash = hashlib.sha1(str(data)).hexdigest()
                    
                    f_name = '_'.join([self.p_type, file_id, f_hash])
                    with open(os.path.join( self.out_dir, f_name), 'wb') as f:
                        logging.debug("%s: Packer writing <%s>" % (self.name, f_name))
                        np.save(f, data)
                    mess['f_name'] = f_name
                    #data.fill(0) 
                    self.release_data()
                   
                    self.events['data_ready'].clear()
                    self.events['add_data'].set()
                    self.out_q.put(mess)
                    logging.debug("%s: file_id<%s> processed" % (self.name, file_id))
                    wait_ctr = 0
                except Empty:
                    logging.debug("%s: waiting on file_id" % self.name)
                    wait_ctr += 1
                    if wait_ctr > 10:
                        logging.error("%s: Packer has data but no file_id, irrecoverable" % self.name)
                        raise Exception("%s: Packer has data but no file_id, irrecoverable" % self.name)
            elif self.events['die'].is_set():
                self.events['die'].clear()
                logging.info("%s: exiting..." % self.name)
                return

    def release_data(self):
        for m in self.shared_mem.itervalues():
            l = m.get_lock()
            l.release()
       
    def get_data(self):
        """
        Returns the np array wrapping the shared memory
        Note: when done with the data, you must call release_data()
            a lock on the shared memory is acquired
        """
        shared_mem = self.shared_mem
        for m in shared_mem.itervalues():
            l = m.get_lock()
            l.acquire()
        myshape = np.frombuffer(shared_mem['shape'].get_obj(),dtype=int)
        t_shape = []
        N = 1
        for i in myshape:
            if i > 0:
                t_shape.append(i)
                N = N * i
        t_shape = tuple(t_shape)
        
        #Note: this is not a copy, it is a view
        #test with np.may_share_memory or data.ctypes.data
        data =  np.frombuffer(shared_mem['data'].get_obj(), dtype=dtypes.nd_list[shared_mem['dtype'].value])
        data = data[:N]
        data = data.reshape(t_shape)
        return data


class PackerBoss:
    """
        base_name - name for this packer
        //in_q - file_id of current data, local
        out_q - queue containing completed info
        out_dir - place to save completed data
        data_settings - (buffer_size, dtype)
    """
    def __init__(self, base_name,  out_q, out_dir, data_settings):
        self.in_q = Queue()
        self.out_q = out_q
        self.out_dir = out_dir
        self.data_settings = data_settings
        b_size, dtype = data_settings
        self.name = base_name
        self.packer = self._create_packer( 'p_' + base_name, b_size, dtype )
    
    def start(self):
        logging.debug("%s: starting packer.." %(self.name))
        self.packer.start()

    def kill(self):
        logging.debug("%s: killing packer.." %(self.name))
        self.packer.die()

    def ready(self):
        return self.packer.events['add_data'].is_set()

    def get_mem(self):
        self.packer.events['add_data'].clear()
        return self.packer.get_mem()

    def set_meta(self, file_id, shape, dtype):
        self.in_q.put(file_id)
        self.packer.set_meta(shape, dtype)

    def release(self):
        self.packer.release_mem()
        self.packer.events['data_ready'].set()

    def _create_packer(self, name, dsize, dtype, p_type='rms'):
        
        logging.debug("%s: creating packer.." %(self.name))
        sm = self._create_shared( dsize, dtype)
        ev = self._create_events()
        p = Packer(name,p_type, self.in_q, self.out_q, sm,ev, self.out_dir)
        ps = PackerStruct('ps_'+name, sm, ev, self.in_q,self.out_q, process=p)
        return ps

    def _create_shared(self, dsize, dtype):
        logging.debug("%s: creating shared_memory.." %(self.name))
        shared_mem = {}
        shared_mem['data'] = Array(dtypes.to_ctypes(dtype),dsize )
        with shared_mem['data'].get_lock():
            temp = np.frombuffer(shared_mem['data'].get_obj(), dtype)
            temp.fill(0)
        shared_mem['shape'] = Array(dtypes.to_ctypes(np.int64), 2)
        with shared_mem['shape'].get_lock():
            temp = np.frombuffer(shared_mem['shape'].get_obj(), np.int64)
            temp.fill(0)

        shared_mem['dtype'] = Value('i',dtypes.nd_dict[np.dtype(dtype)])
        return shared_mem

    def _create_events(self):
        events = {}
        events['add_data'] = Event()
        events['data_ready'] = Event()
        events['die'] = Event()
        return events

    def clean_up(self):
        logging.debug("%s: cleaning up." %(self.name))
        ctr = 0
        while self.packer.process.is_alive() and ctr < 5:
            time.sleep(1)
            ctr += 1
        if self.packer.process.is_alive():
            logging.debug("%s: killing the hard way" %(self.name))
            self.packer.process.terminate()
        self.packer.process.join()
        logging.debug("%s: This house is clean" %(self.name))


class PackerStruct:
    """
    Simple ds for interacting with Packer
    """
    def __init__(self, name,shared_mem, events, in_q,out_q, process=None):
        self.name = name
        self.shared_mem = shared_mem
        self.in_q = in_q
        self.out_q = out_q
        self.events = events
        self.process = process

    def start(self):
        for e in self.events.itervalues():
            e.clear()
        
        self.process.start()

    def die(self):
        self.events['die'].set()

    def get_mem(self):
        self.data_lock = self.shared_mem['data'].get_lock()
        self.data_lock.acquire()
        return self.shared_mem['data']

    def set_meta(self, shape, dtype):
        with self.shared_mem['shape'].get_lock(): 
            self.shared_mem['shape'][:len(shape)] = shape[:]
        with self.shared_mem['dtype'].get_lock():
            self.shared_mem['dtype'].value = dtypes.nd_dict[np.dtype(dtype)]

    def release_mem(self):
        self.data_lock.release()




