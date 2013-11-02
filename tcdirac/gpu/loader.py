import sys
import errno
import time
import os
import os.path
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
from gpu import processes


class LoaderBoss:
    def __init__(self, base_name, file_q,indir,data_settings):
        self.file_q = file_q
        self.indir = indir
        self.base_name = base_name
        self.data_settings = data_settings
        self.loaders = self._createLoaders('_'.join(['proc',base_name]), data_settings)
        self.loader_dist = self._createLoaderDist()

    def empty(self):
        """
        Returns true if no new data and all present data has been used
        """
        a_loader=self.loaders[self.data_settings[0][0]]
        return self.file_q.empty() and a_loader.q.empty() and a_loader.events['add_data'].is_set() and not a_loader.events['data_ready'].is_set()

    def _createLoaderDist(self):
        self._ld_die_evt = Event()
        return LoaderDist( '_'.join([self.base_name,'ld']), self.file_q,  self.loaders, self._ld_die_evt)

    def _createLoaders(self, base_name, data_settings):
        loaders = {}
        for name, dsize, dtype in data_settings:
            loaders[name] = self._createLoader( '_'.join([base_name,name]), dsize, dtype )
        return loaders


    def start(self):
        for l in self.loaders.itervalues():
            l.start()
        self.loader_dist.start()

    def killAll(self):
        logging.debug("%s: Killing subprocesses"%(self.base_name))
        for l in self.loaders.itervalues():
            l.die()
            l.process.join()
        logging.debug("%s: loaders joined"%(self.base_name))
        self._ld_die_evt.set()
        self.loader_dist.join()
        logging.debug("%s: loader_dist joined"%(self.base_name))
        


    def _createLoader(self,name, dsize, dtype):
        smem = self._create_shared(dsize, dtype)
        evts = self._create_events()
        file_q = Queue()
        l = Loader(file_q, evts, smem, self.indir, name)
        ls = LoaderStruct(name,smem,evts,file_q, process=l)
        return ls

    def _create_shared(self, dsize, dtype):
        shared_mem = {}
        shared_mem['data'] = Array(dtypes.to_ctypes(dtype),dsize )
        shared_mem['shape'] = Array(dtypes.to_ctypes(np.int64), 2)
        shared_mem['dtype'] = Value('i',dtypes.nd_dict[np.dtype(dtype)])
        return shared_mem

    def _create_events(self):
        events = {}
        events['add_data'] = Event()
        events['data_ready'] = Event()
        events['die'] = Event()
        return events

    def set_add_data(self):
        for v in self.loaders.itervalues():
            v.events['add_data'].set()

    def clear_data_ready(self):
        for v in self.loaders.itervalues():
            v.events['data_ready'].clear()

    def wait_data_ready(self):
        ready = True
        for v in self.loaders.itervalues():
            if not v.events['data_ready'].wait(10):
                ready = False
        if not ready:
            self.killAll()
            return False
        else:
            return True



class LoaderStruct:
    def __init__(self,name,shared_mem,events,file_q,process=None):
        self.name = name
        self.shared_mem = shared_mem
        self.events = events
        self.q = file_q
        self.process = process

    def start(self):
        for e in self.events.itervalues():
            e.clear()
        self.process.start()

    def die(self):
        self.events['die'].set()

class LoaderDist(Process):
    """
    takes in data from a single q and distributes it to loaders
    """
    def __init__(self, name, in_q, loaders, evt_death):
        Process.__init__(self, name=name)
        self.in_q = in_q
        self.loaders = loaders
        self.proto_q = loaders[loaders.keys()[0]].q
        self.evt_death= evt_death

    def run(self):
        logging.debug("%s: starting..."% self.name)
        while not self.evt_death.is_set():
            try:
                if self.proto_q.qsize() < 10 + random.randint(2,10):
                    f_name = self.in_q.get(True, 10)
                    logging.debug("%s: distributing <%s>" % ( self.name, f_name) )
                    for k,v in self.loaders.iteritems():
                        v.q.put("%s_%s" % (k, f_name))
                else:
                    logging.debug("%s: sleeping due to full q"%self.name)
                    time.sleep(1)
            except Empty:#thrown by in_#thrown by in_qq
                logging.debug("%s: starving..."%self.name)
                pass
        logging.info("%s: exiting..." % (self.name,))
        
    

class Loader(Process):
    def __init__(self, inst_q, events, shared_mem, indir, name, add_data_timeout=10, inst_q_timeout=3):
        """
            inst_q mp queue that tells process next file name
            events = dict of mp events
                events['add_data'] mp event, true means add data 
                events['data_ready'] mp event, true means data is ready to be consumed
                events['die'] event with instructions to die
            shared_mem = dict containing shared memory
                shared_mem['data'] = shared memory for numpy array buffer
                shared_mem['shape'] = shared memory for np array shape
                shared_mem['dtype'] = shared memory for np array dtype
            indir = string encoding location where incoming np data is being written and should be read from
            name = process name(up to you to make it unique)
            add_data_timeout = time in seconds before you check for death when waiting for gpu to release memory
            inst_q_timeout = time in seconds before you check for death when waiting for new filename
        """
        Process.__init__(self, name=name)
        
        self.instruction_queue = inst_q
        self.smem_data = shared_mem['data']
        self.smem_shape = shared_mem['shape']
        self.smem_dtype = shared_mem['dtype']
        self.evt_add_data = events['add_data']
        self.evt_data_ready = events['data_ready']
        self.evt_die = events['die']
        self.indir = indir
        self._ad_timeout = add_data_timeout
        self._iq_timeout = inst_q_timeout


    def loadMem(self, np_array):
        """
        loads data from a numpy array into the provided shared memory
        """
        shape = np_array.shape
        dt_id = dtypes.nd_dict[np_array.dtype]
        size = np_array.size
        np_array = np_array.reshape(size)

        logging.debug("%s: writing to shared memory" % (self.name,))  
        with self.smem_data.get_lock():
            self.smem_data[:size] = np_array[:]
        with self.smem_shape.get_lock():
            self.smem_shape[:len(shape)] = shape[:]
        with self.smem_dtype.get_lock():
            self.smem_dtype.value = dt_id
        logging.debug("%s: shared memory copy complete" % (self.name,))  

    def getData(self, fname):
        return np.load(os.path.join(self.indir, fname))

    def run(self):  
        logging.info("%s: Starting " % (self.name,)) 
        old_md5 = '0'
        #get file name for import
        fname = self.instruction_queue.get()
        new_md5 = self.getMD5(fname)
        logging.debug("%s: loading file <%s>" %(self.name, fname))
        data = self.getData(fname)
        logging.debug("%s: <%s> loaded %f MB " % (self.name, fname, data.nbytes/1048576.0))
        while True:
            self.evt_add_data.wait(self._ad_timeout)
            if self.evt_add_data.is_set():
                logging.debug("%s: loading data into mem " % (self.name,)) 
                self.loadMem( data ) 
                logging.debug("%s: clearing evt_add_data"  % (self.name, ))
                self.evt_add_data.clear()
                logging.debug("%s: setting evt_data ready"  % (self.name, ))
                self.evt_data_ready.set()
                logging.debug("%s: getting new file " % (self.name, )) 
                fname = None
                while fname is None:
                    try:
                        fname = self.instruction_queue.get(True, self._iq_timeout) 
                    except Empty:
                        logging.debug("%s: fname timed out " % (self.name, ))
                        if self.evt_die.is_set():
                            logging.info("%s: exiting... " % (self.name,) )  
                            return
                logging.debug("%s: new file <%s>" %(self.name, fname))
                old_md5 = new_md5
                new_md5 = self.getMD5(fname)
                if new_md5 != old_md5:
                    data = self.getData(fname)
                    logging.debug("%s: <%s> loaded %f MB " % (self.name, fname, data.nbytes/1048576.0))
                else:
                    logging.debug("%s: same data, recycle reduce reuse"  % (self.name, ))
            elif self.evt_die.is_set():
                logging.info("%s: exiting... " % (self.name,) )  
                return

    def getMD5(self, fname):
        """
        Given a formatted filename, returns the precalculated md5 (really any kind of tag)
        """
        return fname.split('_')[-1]
            
def testLoader(pid=0):
    bdir = '/scratch/sgeadmin/'
    np_list = []
    inst_q = Queue()
    data_settings = [('exp', 200*20000, np.float32), ('sm',5*200, np.uint32)]
    
    """
            events = dict of mp events
                events['add_data'] mp event, true means add data 
                events['data_ready'] mp event, true means data is ready to be consumed
                events['die'] event with instructions to die
            shared_mem = dict containing shared memory
                shared_mem['data'] = shared memory for numpy array buffer
                shared_mem['shape'] = shared memory for np array shape
                shared_mem['dtype'] = shared memory for np array dtype
    """

    for i in range(100):
        a = np.random.rand(20,50).astype(np.float)
        f_hash = hashlib.sha1(a).hexdigest()
        base = '_'.join( [  str(random.randint(1000,10000)), f_hash] )
        orig = {}
        for n, _, dtype in data_settings:
            a = np.random.rand(20,50).astype(dtype)
            fname = '_'.join([n, base])
            orig[n] = a
            with open(os.path.join(bdir,fname),'wb') as f:
                np.save(f,a)
        inst_q.put(base)
        inst_q.put(base)
        np_list.append(orig)
    
    db = LoaderBoss(str(pid),inst_q,bdir,data_settings)
    db.start()
    db.set_add_data()
    for a in np_list:
        for i in range(2):
            db.wait_data_ready()                   
            db.clear_data_ready()
            for k,v in a.iteritems():
                ml = db.loaders[k]
                

                myshape = np.frombuffer(ml.shared_mem['shape'].get_obj(),dtype=int)
                size = myshape[0]*myshape[1]
                a_copy =  np.frombuffer(ml.shared_mem['data'].get_obj(), dtype=dtypes.nd_list[ml.shared_mem['dtype'].value])
                a_copy = a_copy[:size]
                a_copy = a_copy.reshape((myshape[0],myshape[1]))
                
                logging.info( "Tester: Copy Matches %s" % (str(np.allclose(a[k], a_copy)),))
            db.set_add_data()
            
    while not db.empty():
        
        time.sleep(.5)
    logging.info( "Tester: no data, all processed, killing sp")
    db.killAll()
    logging.info( "Tester: exitted gracefully")

if __name__ == "__main__":
    sys.path.append('/home/sgeadmin/hdproject/tcDirac/tcdirac')
    print sys.path
    tcdirac.debug.initLogging("tcdirac_gpu_mptest.log", logging.DEBUG)
    sendLogSO()
    temp = []
    for i in range(1):
        p = Process(target=testLoader, args=(i,))
        temp.append(p)
        p.start()
    for p in temp:
        p.join()
