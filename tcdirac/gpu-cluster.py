import errno
import logging
import os
import os.path
import boto
from boto.s3.key import Key
import logging
from boto.sqs.connection import SQSConnection
import cPickle
import time
from multiprocessing import Process, Queue, Lock, Value
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
            print "Shite ", e
            #clean up processes and resources, then raise error 
            self._kill_processes()
            self._join_processes(3)
            ctx.pop()
            raise
        print "Graceful Death"
        self._kill_processes()
        self._join_processes()
        ctx.pop()
        print "Outa here"

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
        print "senging exit actions"
        for i in range(self.num_processes):
            self.q_g2p.put({'action':'exit'})           
        logging.debug("Death warrants sent")
        self._pause = 0
        while self.num_processes > 0:
            try:
                print "Waiting for grace"
                while True:
                    try:
                        data = self.q_p2g.get(block=True, timeout=5)
                        
                        break
                    except IOError, e:
                        print "IOERRor"
                        if e.errno != errno.EINTR:
                            raise
                if 'msg' in data and  data['msg'] == 'exiting':
                    logging.debug("Subprocess exitted")
                    self.num_processes -= 1
                else:
                    logging.warning("Master: Killing Subprocess, but recvd non death msg [%s]" % str(data) )
            except Empty:
                #Tried to play nice, now we just terminate the slackers
                print "Fuck it"
                self._hard_terminate()
                self.num_processes = 0
        self._join_processes( 3 )

    def _hard_terminate(self):
        for p in self._processes:
            if p.is_alive():
                print "DIE"
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
    

    g = GPUDirac(world_comm, sqs_d2g, sqs_g2d, source_bucket, dest_bucket,dev_id=0, num_processes=5)
    g.run()

                
                
        
