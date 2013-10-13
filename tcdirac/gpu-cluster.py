import logging
import os
import os.path
import boto
from boto.s3.key import Key
import logging
from boto.sqs.connection import SQSConnection
import cPickle
import time
from multiprocessing import Process, Queue, Lock
import numpy as np
from boto.sqs.message import Message
from mpi4py import MPI
from Queue import Empty
import random
import pandas
import itertools
import scipy.misc
from tempfile import TemporaryFile

from gpu import processes

import pycuda.driver as cuda

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
    def __init__(self, sqs_d2g, sqs_g2d, source_bucket, dest_bucket,q_p2g,q_g2p,stout_lock, max_qsize=10):
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
        print data
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
            print "Action from gpu", results['action']
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
            print "No results"
            self._pause += 1
            logging.debug( "DW: no mess from gpu" )
        return True

    def _handlePause(self):
        self._tt.start_wait()
        if self._pause > 0:
            print "DW start sleep"
            logging.debug( "DW: sleeping")
            time.sleep((1.1 ** self._pause) + self._pause + random.random())
            print "DW end sleep"
        self._tt.end_wait()

    def _putData(self, data):
        print "putting data"
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
        print "Queue Att", self.sqs_g2d.get_attributes()
        self.sqs_g2d.write(m)
        self._tt.end_work()




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

    def run(self):
        self._createProcesses()
        cuda.init()
        dev = cuda.Device(self._dev_id)
        
        ctx = dev.make_context()
        try:
            self.consume()
        except Exception as e:
            print e
            self._kill_processes()
            while self.num_processes:
                self.q_p2g.get()
            raise
        else:
            self._joinProcesses()
            ctx.pop()

    def _createProcesses(self):
        self.stout_lock = Lock()
        logging.info("Starting DataWorkers")
        param = (self.sqs_d2g, self.sqs_g2d, self.source_bucket, self.dest_bucket,self.q_p2g,self.q_g2p, self.stout_lock )
        my_processes = [DataWorker(*param) for i in range(self.num_processes)]
        for p in my_processes:
            p.start()
        self._processes = my_processes

    def _joinProcesses(self):
        logging.info("Joining DataWorkers")
        for p in self._processes:
            p.join()

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
        while True:
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

            if self._pause:
                if self._pause > 5:
                    logging.info("Exitting due to lack of work")
                    if self.num_processes == 0:
                        logging.debug( "no processes to kill" )
                        return
                    else:
                        return
                time.sleep(self._pause)

    def _kill_processes(self):
        logging.debug( "killing processes" )
        msg = {'action':'exit'}
        for i in range(self.num_processes):
            self.q_g2p.put({'action':'exit'})           
        logging.debug("Death warrants sent")
        self._pause = 0

        
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

        logging.debug("Processing complete")
        self.q_g2p.put(result)
        logging.debug("results sent to DW")       
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
    for i in range(3):
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



if __name__ == "__main__":
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
    

    g = GPUDirac(world_comm, sqs_d2g, sqs_g2d, source_bucket, dest_bucket,dev_id=0, num_processes=2)
    g.run()

                
                
        
