import logging
import os
import os.path
import boto
from boto.s3.key import Key
import logging
from boto.sqs.connection import SQSConnection
import cPickle
import time
from multiprocessing import Process, Queue
import numpy as np
from boto.sqs.message import Message
from mpi4py import MPI
from Queue import Empty
import random
import pandas
import itertools
import scipy.misc
from tempfile import TemporaryFile
class DataWorker(Process):
    def __init__(self, sqs_d2g, sqs_g2d, source_bucket, dest_bucket,q_p2g,q_g2p):
        Process.__init__(self)
        self.sqs_conn = boto.connect_sqs()
        self.s3_conn = boto.connect_s3()
        self.sqs_d2g = self.sqs_conn.create_queue(sqs_d2g)
        self.sqs_g2d = self.sqs_conn.create_queue(sqs_g2d)
        self.source_bucket = self.s3_conn.get_bucket(source_bucket)
        self.dest_bucket = self.s3_conn.get_bucket(dest_bucket)
        self.q_p2g = q_p2g
        self.q_g2p = q_g2p
        self._pause = 0

    def run(self):
        cont = True
        while cont:
            self._handlePause()
            self._handleSQS()
            cont = self._handleResults()




    def _handleData(self, file_name):
        logging.debug("DW: handleData(%s)"%file_name)
        k = Key(self.source_bucket)
        k.key = file_name
        outfile = TemporaryFile()
        k.get_contents_to_file(outfile)
        outfile.seek(0)
        data = np.load(outfile)

        data['action'] = 'process'
        self.q_p2g.put(data)

    def _cleanUp(self):
        self.q_p2g.close()
        self.q_g2p.close()
    
    def _handleSQS(self):
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


    def _handleResults(self):
        try:
            results = self.q_g2p.get(False) 
            logging.debug( "DW: Mess from gpu" )
            if results['action'] == 'exit':
                self.q_p2g.put({'action':'message', 'msg':'exiting'})
                self._cleanUp()
                logging.info("DW: exiting after message from master")
                return False
            elif results['action'] == 'transmit':        
                self._putData(results)
                logging.debug( "DW: Sent mess to sqs and s3" )
                self._pause = 0
        except Empty:
            self._pause += 1
            logging.debug( "DW: no mess from gpu" )
        finally:
            return True

    def _handlePause(self):
        if self._pause > 0:
            logging.debug( "DW: sleeping")
            time.sleep((1.1 ** self._pause) + self._pause + random.random())

    def _putData(self, data):
        logging.debug( "DW: putting Data" )
        fname = data['fname']
        data_s = cPickle.dumps(data)
        k = Key(self.dest_bucket)
        k.key = fname
        k.set_contents_from_string(data_s)
        self.sqs_g2d.put(fname)





class GPU:
    def __init__(self, world_comm, sqs_d2g, sqs_g2d, source_bucket, dest_bucket, num_processes=5):
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

    def run(self):
        self._createProcesses()
        self.consume()
        self._joinProcesses()

    def _createProcesses(self):
        logging.info("Starting DataWorkers")
        param = (self.sqs_d2g, self.sqs_g2d, self.source_bucket, self.dest_bucket,self.q_p2g,self. q_g2p)
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
        nothing = 0
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
                nothing += 1

            if nothing:
                print "nothing"
                if nothing > 5:
                    logging.info("Exitting due to lack of work")
                    if self.num_processes == 0:
                        logging.debug( "no processes to kill" )
                        return
                    else:
                        logging.debug( "killing processes" )
                        self.sqs_conn = boto.connect_sqs()
                        sqs_d2g = self.sqs_conn.create_queue(self.sqs_d2g)
                        for i in range(self.num_processes):
                            m = Message()
                            m.set_body('quit')
                            status = sqs_d2g.write(m)
                        logging.debug("Death warrants sent")
                        return
                time.sleep(2*nothing)

        
            
        

def seed(sqs_d2g = 'tcdirac-togpu-00',
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
        
        fname = "afile_%i_%i"%(i, random.randint(1,99999))
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
        n_size = random.randint(5,300)

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
    data['expression_matrix'] = None#expression_matrix
    data['gene_map'] = None#gene_map
    data['sample_map'] = sample_map
    data['network_map'] = None#network_map
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
    seed()
    """
    world_comm = MPI.COMM_WORLD
    initLogging("tcdirac_gpu_mpi_%i.log"%world_comm.rank, logging.DEBUG)
    

    g = GPU(world_comm, sqs_d2g, sqs_g2d, source_bucket, dest_bucket,2)
    g.run()"""

                
                
        
