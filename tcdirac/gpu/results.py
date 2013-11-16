
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
from loader import MaxDepth 

class PackerQueue:
    """
    Object containing a list of PackerBosses for the gpu to write to
    """
    def __init__(self, name, results_q, out_dir, data_settings):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.results_q = results_q #queue containing meta information
        self.out_dir = out_dir
        self.data_settings = data_settings
        self._bosses = []   
        self._bosses_skip = []
        self._curr = -1
        self.check_out_dir()

    def add_packer_boss(self, num=1):
        if num <= 0:
            return
        else:
            self._bosses.append( PackerBoss( 'pb_' + str(len(self._bosses)), self.results_q,self.out_dir, self.data_settings) )
            self._bosses_skip.append(0)
            self._bosses[-1].start()
            self.add_packer_boss( num - 1)

    def next_packer_boss(self, time_out=0.1, max_depth=None):
        if len(self._bosses) == 0:
            raise Exception("No Loaders")
        if max_depth is None:
            max_depth = 2*len(self._bosses)#default max_depth to 2 passes of the queue
        if max_depth <= 0:
            raise MaxDepth("Max Depth exceeded")

        self._curr = (self._curr + 1)%len(self._bosses)
        if self._bosses[self._curr].ready():
            self._bosses_skip[self._curr] = 0
            return self._bosses[self._curr]
        else:
            if self._bosses_skip[self._curr] > 0:
                time.sleep(.1)
            self._bosses_skip[self._curr] += 1
            return self.next_packer_boss(time_out, max_depth=max_depth-1)

    def checkSkip(self, max_skip=3):
        over_limit = [i for i,l in enumerate(self._bosses_skip) if l > max_skip]
        temp = []
        for i in over_limit:
            temp.append(self._bosses[i])
            self._bosses[i].kill_all()
        for i in over_limit:
            self._bosses[i] =  LoaderBoss( 'lb_' + str(i), self.results_q, self.data_settings)
            self._bosses_skip[i] = 0
        for l in temp:
            l.clean_up()

    def check_out_dir(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            self.logger.info("PackerQueue: dir did not exist, created %s" % (self.out_dir))

    def remove_packer_boss(self):
        if len(self._bosses) <=0:
            raise Exception("Attempt to remove Loader from empty LoaderQueue")
        temp = self._bosses[-1]
        temp.kill_all()
        self._bosses = self._bosses[:-1]
        self._bosses_skip = self._bosses_skip[:-1]
        self._curr = self._curr%len(self._bosses)
        temp.clean_up()

    def no_data(self):
        return self.results_q.empty()

    def kill_all(self):
        for l in self._bosses:
            l.kill()
            self.logger.debug( "%s you killed my father, prepared to die" % l.name)
        for l in self._bosses:
            l.clean_up()
        self._bosses = []
        self._bosses_skip = []
        self._curr = -1

    def set_data_settings(self, data_settings):
        self.data_settings = data_settings

    def num_sub(self):
        count = 0
        for p in self._bosses:
            if p.is_alive():
                count += 1
        return count
            

class Packer(Process):
    def __init__(self, name,p_type, in_q, out_q, smem,events, out_dir, dr_timeout=10):
        Process.__init__(self, name=name)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.in_q = in_q
        self.out_q = out_q
        self.shared_mem = smem
        self.events = events
        self.out_dir = out_dir
        self._dr_timeout = dr_timeout
        self.p_type = p_type
        self.daemon = True


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
                        self.logger.debug("%s: Packer writing <%s>" % (self.name, f_name))
                        np.save(f, data)
                    mess['f_name'] = f_name
                    #data.fill(0) 
                    self.release_data()
                   
                    self.events['data_ready'].clear()
                    self.events['add_data'].set()
                    self.out_q.put(mess)
                    self.logger.debug(" file_id<%s> processed" % (file_id))
                    wait_ctr = 0
                except Empty:
                    self.logger.debug("waiting on file_id")
                    wait_ctr += 1
                    if wait_ctr > 10:
                        self.logger.error(" Packer has data but no file_id, irrecoverable")
                        raise Exception("%s: Packer has data but no file_id, irrecoverable" % self.name)
            elif self.events['die'].is_set():
                self.events['die'].clear()
                self.logger.info(" exiting...")
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
        self.name = base_name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.in_q = Queue()
        self.out_q = out_q
        self.out_dir = out_dir
        self.data_settings = data_settings
        self.packer = self._create_packer( 'p_' + base_name )
    
    def start(self):
        self.logger.debug(" starting packer..")
        self.packer.start()

    def kill(self):
        self.logger.debug(" killing packer..")
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

    def _create_packer(self, name,  p_type='rms'):
        #only need dsize and dtype, but want to make consistent
        #with data_settings for loader
        for ds in self.data_settings:
            rms, dsize, dtype = ds
        self.logger.debug(" creating packer..")
        sm = self._create_shared( dsize, dtype)
        ev = self._create_events()
        p = Packer(name,p_type, self.in_q, self.out_q, sm,ev, self.out_dir)
        ps = PackerStruct('ps_'+name, sm, ev, self.in_q,self.out_q, process=p)
        return ps

    def _create_shared(self, dsize, dtype):
        self.logger.debug(" creating shared_memory..")
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
        self.logger.debug(" cleaning up.")
        ctr = 0
        while self.packer.process.is_alive() and ctr < 5:
            time.sleep(1)
            ctr += 1
        if self.packer.process.is_alive():
            self.logger.debug(" killing the hard way")
            self.packer.process.terminate()
        self.packer.process.join()
        self.logger.debug(" This house is clean")

    def is_alive(self):
        return self.packer.process.is_alive()



class PackerStruct:
    """
    Simple ds for interacting with Packer
    """
    def __init__(self, name,shared_mem, events, in_q,out_q, process=None):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
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
