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

class Loader(Process):
    def __init__(self, inst_q,evt_add_data, evt_data_ready,evt_die, smem_data, smem_shape, smem_dtype, indir, name, add_data_timeout=10, inst_q_timeout=3):
        """
            inst_q mp queue that tells process next file name
            evt_add_data mp event, true means add data 
            evt_data_ready mp event, true means data is ready to be consumed
            smem_data = shared memory for numpy array buffer
            smem_shape = shared memory for np array shape
            smem_dtype = shared memory for np array dtype
            indir = string encoding location where incoming np data is being written
            name = process name
            add_data_timeout = time in seconds before you check for death when waiting for gpu to release memory
            inst_q_timeout = time in seconds before you check for death when waiting for new filename
        """
        Process.__init__(self, name=name)
        self.instruction_queue = inst_q
        self.smem_data = smem_data
        self.smem_shape = smem_shape
        self.smem_dtype = smem_dtype
        self.evt_add_data = evt_add_data
        self.evt_data_ready = evt_data_ready
        self.evt_die = evt_die
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
        while self.evt_add_data.wait(self._ad_timeout):
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
            
def testLoader():
    dtype_wrapper = [(np.float32, ctypes.c_float), (np.float, ctypes.c_double), (np.uint32, ctypes.c_uint), (np.uint, ctypes.c_ulonglong)]
    bdir = '/scratch/sgeadmin/'
    np_list = []
    inst_q = Queue()
    
    for i in range(10):
        a = np.random.rand(20,50).astype(np.float32)
        f_hash = hashlib.sha1(a).hexdigest()
        fname = '_'.join(['mf', str(random.randint(1000,10000)), f_hash])
        with open(os.path.join(bdir,fname),'wb') as f:
            np.save(f,a)

        inst_q.put(fname)
        inst_q.put(fname)
        np_list.append(a)
        
    evt_add_data = Event()
    evt_add_data.clear()
    evt_data_ready = Event()
    evt_data_ready.clear()
    evt_die = Event()
    evt_die.clear()
    
    smem_data = Array(ctypes.c_float, a.size *4)
    smem_shape = Array(ctypes.c_longlong, 2)
    smem_dtype = Value('i',0)
    indir = bdir
    print "starting"
    l = Loader( inst_q,evt_add_data, evt_data_ready,evt_die, smem_data, smem_shape, smem_dtype, indir, name="testname")
    l.start()
    evt_add_data.set()
    for a in np_list:
        for i in range(2):        
            print "setting data"
            print "evt_add_data", evt_add_data.is_set()
            print "set"
            
            print "evt_data_ready", evt_data_ready.is_set()
            evt_data_ready.wait()
            evt_data_ready.clear()
            myshape = np.frombuffer(smem_shape.get_obj(),dtype=int)
            size = myshape[0]*myshape[1]
            a_copy =  np.frombuffer(smem_data.get_obj(), dtype=dtypes.nd_list[smem_dtype.value])
            evt_add_data.set() 
            a_copy = a_copy[:size]
            a_copy = a_copy.reshape((myshape[0],myshape[1]))

            print "Matches", np.allclose(a, a_copy)

    evt_die.set()
    time.sleep(15)
    l.join()
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

    testLoader()
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

                
                
        
