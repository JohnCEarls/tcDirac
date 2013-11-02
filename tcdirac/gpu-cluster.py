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

from mpi4py import MPI

import pycuda.driver as cuda

import dtypes
from gpu import processes


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

class DataWorker(Process):
    def __init__(self,p_rank,  sqs_d2g, sqs_g2d, source_bucket, dest_bucket,q_p2g,q_g2p,stout_lock, max_qsize):
        Process.__init__(self)
        self.sqs_conn = boto.connect_sqs()
        self.s3_conn = boto.connect_s3()
       
        self.sqs_d2g = self.sqs_conn.create_queue(sqs_d2g)
        self.sqs_g2d = self.sqs_conn.create_queue(sqs_g2d)
        self.source_bucket = self.s3_conn.get_bucket(source_bucket)
        self.dest_bucket = self.s3_conn.get_bucket(dest_bucket)
        self.q_p2g = q_p2g
        self.q_g2p = q_g2p
        self._max_qsize = max_qsize
        self._pause = 0
        self._tt = TimeTracker()       
        self.stout_lock = stout_lock
        self._p_rank = p_rank

    def run(self):
        cont = True
        while cont:
            self._handlePause()
            if self.q_p2g.qsize() < self._max_qsize:
                self._handleSQS()
            else:
                self._pause += 1
            cont = self._handleResults()
        self.stout_lock.acquire()
        print "DW"
        self._tt.print_stats()
        self.stout_lock.release()



    def _handleData(self, file_name):
        self._tt.start_work()
        logging.debug("DW: handleData(%s)"%file_name)
        k = Key(self.source_bucket)
        k.key = file_name
        outfile = TemporaryFile()
        k.get_contents_to_file(outfile)
        outfile.seek(0)
        data = np.load(outfile)
        msg = {}
        msg['action'] = 'process'
        msg['fname'] = file_name
        for k,v in data.iteritems():
            msg[k] = v
        
        self.q_p2g.put(msg)
        print "DW:put message"
        self._tt.end_work()

    def _cleanUp(self):
        self.q_p2g.close()
        self.q_g2p.close()
    
    def _handleSQS(self):
        
        self._tt.start_work()
        logging.debug( "DW: getting Messages" )
        messages = self.sqs_d2g.get_messages()
        logging.debug( "DW: mess from SQS")
        self._pause += 1
        for mess_sqs in messages:
            mess_dict = cPickle.loads(mess_sqs.get_body())
            if mess_dict['action'] == 'quit':
                logging.debug("DW: QUIT message recvd")
                #send message to master, cause you are not the boss of me
                self.q_p2g.put({'action':'message', 'mess':'QUIT RECVD'})
            elif mess_dict['action'] == 'process':
                self._handleData(mess_dict['file_name'])
                self.sqs_d2g.delete_message(mess_sqs)
                self._pause = 0

        self._tt.end_work()

    def _handleResults(self):
        try:
            
            self._tt.start_work()
            results = self.q_g2p.get(False) 
            logging.debug( "DW: Mess from gpu" )
            if results['action'] == 'exit':
                self.q_p2g.put({'action':'message', 'msg':'exiting'})
                self._cleanUp()
                print "DW:Acknowledging exit"
                logging.info("DW: exiting after message from master")
                return False
            elif results['action'] == 'transmit':        
                self._putData(results)
                logging.debug( "DW: Sent mess to sqs and s3" )
                self._pause = 0
            self._tt.end_work()
        except Empty:
            self._pause += 1
            logging.debug( "DW: no mess from gpu" )
        return True

    def _handlePause(self):
        self._tt.start_wait()
        if self._pause > 0:
            logging.debug( "DW: sleeping")
            sleep_time = (1.1 ** self._pause) + self._pause + random.random()
            sleep_time = min( sleep_time, 5)
            time.sleep(sleep_time)
        self._tt.end_wait()

    def _putData(self, data):
        self._tt.start_work()
        logging.debug( "DW: putting Data" )
        fname = data['fname']
        data_s = cPickle.dumps(data)
        k = Key(self.dest_bucket)
        k.key = fname
        k.set_contents_from_string(data_s)
        m = Message()
        ms = {}
        ms['file_name'] = fname
        ms['action'] = 'results'
        m.set_body(cPickle.dumps(ms))
        self.sqs_g2d.write(m)
        self._tt.end_work()


class LoaderBoss:
    def __init__(self, base_name, file_q,indir,data_settings):
        self.file_q = file_q
        self.indir = indir
        self.base_name = base_name
        self.loaders = self._createLoaders('_'.join(['proc',base_name]), data_settings)
        self.loader_dist = self._createLoaderDist()


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
        print "setting add data"
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
            print "data ready"
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
        print "boppity",self.evt_add_data.is_set()
        print "to", self._ad_timeout
        while True:
            self.evt_add_data.wait(self._ad_timeout)
            print "bibity", self.evt_add_data.is_set()
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
        print "outa_here"

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
    print "shite"
    db.set_add_data()
    for a in np_list:
        for i in range(2):
            db.wait_data_ready()                   
            print "got data ready"
            db.clear_data_ready()
            print "clearing data ready"
            for k,v in a.iteritems():
                print "a"
                ml = db.loaders[k]
                

                myshape = np.frombuffer(ml.shared_mem['shape'].get_obj(),dtype=int)
                size = myshape[0]*myshape[1]
                a_copy =  np.frombuffer(ml.shared_mem['data'].get_obj(), dtype=dtypes.nd_list[ml.shared_mem['dtype'].value])
                a_copy = a_copy[:size]
                a_copy = a_copy.reshape((myshape[0],myshape[1]))
                
                print "Matches", np.allclose(a[k], a_copy)
            db.set_add_data()

    db.killAll()
    print "exitted gracefully"


class GPUBase:
    def __init__(self, world_comm, sqs_d2g, sqs_g2d, source_bucket, dest_bucket, dev_id, num_processes=2):
        #SQS queues
        self.sqs_d2g = sqs_d2g
        self.sqs_g2d = sqs_g2d
        #s3 bucket locations
        self.source_bucket = source_bucket
        self.dest_bucket = dest_bucket
        #subprocess queues
        self.q_p2g = Queue() 
        self.q_g2p = Queue()

        self.num_processes = num_processes
        self._pause = 0
        self._dev_id = dev_id
        self._max_pause = 15
        self._max_queue_size = Value('i', 10)

    def run(self):
        self._createProcesses()
        cuda.init()
        dev = cuda.Device(self._dev_id)
        
        ctx = dev.make_context()
        try:
            self.consume()
        except Exception as e:
            logging.error(type(e) + ' - ' + str(e))
            logging.error("Master: Clean up processes and resources, then raise error")
            self._kill_processes()
            self._join_processes(3)
            ctx.pop()
            raise
        logging.debug("Master: Graceful Death")
        self._kill_processes()
        self._join_processes()
        ctx.pop()
        logging.debug("Master: gpuBase.run() exitting")

    def _createProcesses(self):
        self.stout_lock = Lock()
        logging.info("Starting DataWorkers")
        param = [self.sqs_d2g, self.sqs_g2d, self.source_bucket, self.dest_bucket,self.q_p2g,self.q_g2p, self.stout_lock]
        my_processes = [DataWorker(**self._get_process_args(i)) for i in range(self.num_processes)]
        for p in my_processes:
            p.start()
        self._processes = my_processes

    def _get_process_args(self, proc_id ):
        """
        (p_rank,  sqs_d2g, sqs_g2d, source_bucket, dest_bucket,q_p2g,q_g2p,stout_lock, max_qsize)
        """
        args = {}
        args['p_rank'] = proc_id
        args['sqs_d2g'] = self.sqs_d2g
        args['sqs_g2d'] = self.sqs_g2d
        args['source_bucket'] = self.source_bucket
        args['dest_bucket'] = self.dest_bucket
        args['q_p2g'] = self.q_p2g
        args['q_g2p'] = self.q_g2p
        args['stout_lock'] = self.stout_lock
        args['max_qsize'] = self._max_queue_size
        return args

    def _change_queue_size( self, queue_size ):
        self._max_queue_size = queue_size
        

    def _join_processes(self, timeout=2):
        logging.info("Joining DataWorkers")
        for p in self._processes:
            p.join( timeout )
            if p.is_alive():
                logging.debug("Master: join timed out")

    def _handleMessage(self, msg):
        logging.debug("MSG: %s"% data['msg'])
        if data['msg'] == 'exiting':
            logging.debug("Subprocess exitted")
            self.num_processes -= 1
        if data['msg'] == 'QUIT RECVD':
            for i in range(self.num_processes):
                self.q_g2p.put({'action':'exit'})
        

    def _handleData(self, data):
        
        #TODO do something with the data
        time.sleep(2)


        data['action'] = 'transmit'
        logging.debug("Processing complete")
        self.q_g2p.put(data)
        logging.debug("results sent to DW")       
        self._pause = 0

    def consume(self):
        logging.info( "Starting consumption" )
        cont = True
        while cont:
            logging.debug("Master getting data")
            try:
                data = self.q_p2g.get( False )

                logging.debug( "Master got data" )

                if data['action'] == 'message':
                    self._handleMessage(data)
                elif data['action'] == 'process':
                    self._handleData(data)

            except Empty:
                logging.debug("No data recvd")
                self._pause += 1
            cont = self._handle_pause()

            

    def _handle_pause(self):
        print "_pause", self._pause
        if self._pause > self._max_pause:
            logging.info("Exitting due to lack of work")
            return False
        if self._pause > 2:
            print "increasing queue size"
            self._max_queue_size.value += 1
        time.sleep(self._pause)
        return True
            

    def _kill_processes(self):
        logging.debug( "killing processes" )
        for i in range(self.num_processes):
            self.q_g2p.put({'action':'exit'})           
        logging.debug("Death warrants sent")
        self._pause = 0
        while self.num_processes > 0:
            try:
                while True:
                    try:
                        data = self.q_p2g.get(block=True, timeout=5)
                        
                        break
                    except IOError, e:
                        print "IOERRor"
                        if e.errno != errno.EINTR:
                            raise
                if 'msg' in data and  data['msg'] == 'exiting':
                    logging.debug("Master: a Subprocess messaged exit")
                    self.num_processes -= 1
                else:
                    logging.warning("Master: Killing Subprocess, but recvd non death msg [%s]" % str(data) )
            except Empty:
                #Tried to play nice, now we just terminate the slackers
                logging.error("Master: subprocesses not exitting gracefully")
                self._hard_terminate()
                self.num_processes = 0
        self._join_processes( 3 )

    def _hard_terminate(self):
        for p in self._processes:
            if p.is_alive():
                logging.debug("Master: Process <%i> is still alive. Sending SIGTERM."%p.pid)
                p.terminate()

        
class GPUDirac(GPUBase):
    def _compute(self, data):
        sample_block_size, npairs_block_size, nets_block_size = self._getBlockSize(data)
        expression_matrix = data['expression_matrix']
        gene_map = data['gene_map'] 
        sample_map = data['sample_map']
        network_map = data['network_map']
        print "Computing"
        rt,rt,rms = processes.runDirac( expression_matrix, gene_map, sample_map, network_map, sample_block_size, npairs_block_size, nets_block_size,rms_only=True )
        print "done computing"
        result = {}
        result['fname'] = data['fname']
        result['action'] = 'transmit'
        result['rms'] = rms.res_data
        return result       

    def _handleData(self, data):
        print "handling data" 
        #TODO do something with the data
        result = self._compute(data)
        logging.debug("Master: Processing complete")
        self.q_g2p.put(result)
        logging.debug("Master: results sent to DW")       
        self._pause = 0

    def _getBlockSize(self, data):
        """
        Prob wanna get smarter about this
        """
        return (32,16,4)

def seed(ignore, sqs_d2g = 'tcdirac-togpu-00',
    sqs_g2d = 'tcdirac-fromgpu-00',
    source_bucket = 'tcdirac-togpu-00',
    dest_bucket = 'tcdirac-fromgpu-00'):

    sqs_conn = boto.connect_sqs()
    s3_conn = boto.connect_s3()
    sqs_d2g = sqs_conn.create_queue(sqs_d2g)
    sqs_g2d = sqs_conn.create_queue(sqs_g2d)
    b = s3_conn.get_bucket(source_bucket)
    for i in range(10):
        dp = {}
        print "gendata"
        st = time.time()
        dp = genFakeData(200, 20000)
        
        fname = "afile_%i_%i_%i"%(ignore, i, random.randint(1,99999))
        print st - time.time()
        st = time.time()
        print "pickleData"
        outfile = TemporaryFile()
        np.savez(outfile, **dp )
        outfile.seek(0)
        #data = cPickle.dumps(dp)
        print st - time.time()
        st = time.time()
        print "senddata"
        k = Key(b)
        k.key = fname
        k.set_contents_from_file(outfile)
        print st - time.time()
        st = time.time()
        print "send q"
        m = Message()
        ms = {}
        ms['file_name'] = fname
        ms['action'] = 'process'
        m.set_body(cPickle.dumps(ms))
        sqs_d2g.write(m)
        print st - time.time()
        st = time.time()
        print "sent 1"


def clearSQS(queue_name):
    sqs_conn = boto.connect_sqs()
    sqs = sqs_conn.create_queue(queue_name)
    sqs.clear()

def genFakeData( n, gn):
    neighbors = random.randint(5, 20) 
    nnets = random.randint(50,300)

    samples = map(lambda x:'s%i'%x, range(n))
    genes = map(lambda x:'g%i'%x, range(gn))
    g_d = dict([(gene,i) for i,gene in enumerate(genes)])
    gm_text = []   
    gm_idx = []


    exp = np.random.rand(len(genes),len(samples)).astype(np.float32)
    exp_df = pandas.DataFrame(exp,dtype=float, index=genes, columns=samples)

    net_map = [0]
   
    for i in range(nnets):
        n_size = random.randint(5,100)

        net_map.append(net_map[-1] + scipy.misc.comb(n_size,2, exact=1))
        net = random.sample(genes,n_size)
        for g1,g2 in itertools.combinations(net,2):
            gm_text.append("%s < %s" % (g1,g2))
            gm_idx += [g_d[g1],g_d[g2]]

    #data
    expression_matrix = exp
    gene_map = np.array(gm_idx)
    #print gene_map[:20]
    sample_map = np.random.randint(low=0,high=len(samples), size=(len(samples),neighbors))
    #print sample_map[:,:3]
    network_map = np.array(net_map)

    data = {}
    data['expression_matrix'] = expression_matrix
    data['gene_map'] = gene_map
    data['sample_map'] = sample_map
    data['network_map'] = network_map
    return data

def initLogging(lname,level):
    rank = MPI.COMM_WORLD.rank
    logdir = '/scratch/sgeadmin/logs/'
    if rank == 0:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    else:
        while not os.path.exists(logdir):
            time.sleep(1)

    log_format = '%(asctime)s - %(name)s rank['+str( rank )+']- %(levelname)s - %(message)s'
    logging.basicConfig(filename=os.path.join(logdir,lname), level=level, format=log_format)
    logging.info("Starting")

def sendLogSO():
    root = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

if __name__ == "__main__":
    initLogging("tcdirac_gpu_mptest.log", logging.DEBUG)
    sendLogSO()
    temp = []
    for i in range(10):
        p = Process(target=testLoader, args=(i,))
        temp.append(p)
        p.start()
    for p in temp():
        p.join()
    
    """
    sqs_d2g = 'tcdirac-togpu-00'
    sqs_g2d = 'tcdirac-fromgpu-00'
    source_bucket = 'tcdirac-togpu-00'
    dest_bucket = 'tcdirac-fromgpu-00'
    clearSQS(sqs_d2g)
    clearSQS(sqs_g2d)
    from multiprocessing import Pool
    p = Pool(6)
    p.map(seed, range(6))
    world_comm = MPI.COMM_WORLD
    initLogging("tcdirac_gpu_mpi_%i.log"%world_comm.rank, logging.DEBUG)
    

    g = GPUDirac(world_comm, sqs_d2g, sqs_g2d, source_bucket, dest_bucket,dev_id=0, num_processes=5)
    g.run()"""

                
                
        
