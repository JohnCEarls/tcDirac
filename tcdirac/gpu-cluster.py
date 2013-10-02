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
class DataWorker(Process):
    def __init__(self, inqueue, outqueue, source_bucket, dest_bucket,data_inq,data_outq):
        Process.__init__(self)
        self.sqs_conn = boto.connect_sqs()
        self.s3_conn = boto.connect_s3()
        self.inqueue = self.sqs_conn.create_queue(inqueue)
        self.outqueue = self.sqs_conn.create_queue(outqueue)
        self.source_bucket = self.s3_conn.get_bucket(source_bucket)
        self.dest_bucket = self.s3_conn.get_bucket(dest_bucket)
        self.data_inq = data_inq
        self.data_outq = data_outq



    def getData(self, file_name):
        print "getData"
        k = Key(self.source_bucket)
        k.key = file_name
        data_s=k.get_contents_as_string()
        data = cPickle.loads(data_s)
        self.data_inq.put(data)

    def run(self):
        nothing = 0
        while True:
            print "getthing Messages"
            mess = self.inqueue.get_messages()
            if len(mess) > 0:
                nothing = 0
                if mess.get_body() == 'quit':
                    return
                else:
                    self.getData(mess.get_body())
            else:
                nothing += 1

            try:
                results = self.data_outq.get() 
                self.putData(results)
                nothing = 0
            except Queue.Empty:
                pass
            if nothing:
                time.sleep(2**nothing)
            if nothing > 10:
                return

    def putData(self, data):
        print "putting Data"
        fname = data['fname']
        data_s = cPickle.dumps(data)
        k = Key(self.dest_bucket)
        k.key = fname
        k.set_contents_from_string(data_s)
        self.outqueue.put(fname)





class GPU:
    def __init__(self, world_comm, inqueue, outqueue, source_bucket, dest_bucket, num_processes=5):
        self.inqueue = inqueue
        self.outqueue = outqueue
        self.source_bucket = source_bucket
        self.dest_bucket = dest_bucket
        self.data_inq = Queue() 
        self.data_outq = Queue()
        self.num_processes = num_processes

    def run(self):
        print "running"
        param = (self.inqueue, self.outqueue, self.source_bucket, self.dest_bucket,self.data_inq,self. data_outq)
        my_processes = [DataWorker(*param) for i in range(self.num_processes)]
        for p in my_processes:
            p.start()
        self.consume()
        for p in my_processes:
            p.join()

    def consume(self):
        print "consuming"
        nothing = 0
        while True:
            
            try:
                data = self.data_inq.get()
                self.data_outq.put(data)
                nothing = 0
            except Queue.Empty:
                nothing += 1
            if nothing:
                print "nothing"
                if nothing > 5:
                    
                    self.sqs_conn = boto.connect_sqs()
                    inqueue = self.sqs_conn.create_queue(inqueue)
                    for i in range(20):
                        m = Message()
                        m.set_body('quit')
                        status = inqueue.write(m)
                    print "enough"
                    return
                time.sleep(2**nothing)

        
            
        

def seed(inqueue = 'tcdirac-togpu-00',
    outqueue = 'tcdirac-fromgpu-00',
    source_bucket = 'tcdirac-togpu-00',
    dest_bucket = 'tcdirac-fromgpu-00'):

    sqs_conn = boto.connect_sqs()
    s3_conn = boto.connect_s3()
    inqueue = sqs_conn.create_queue(inqueue)
    outqueue = sqs_conn.create_queue(outqueue)
    for i in range(1000):
        dp = {}
        dp['fname'] = "file_%i_%i"%(i, MPI.COMM_WORLD.rank)
        dp['data'] = np.random.randn((i+20)*20, 1000)
        out = cPickle.dumps(dp)

        k = Key(s3_conn.get_bucket(source_bucket))
        k.key = dp['fname']
        k.set_contents_from_string(out)
        m = Message()
        m.set_body(dp['fname'])
        outqueue.write(m)
    
    

if __name__ == "__main__":
    inqueue = 'tcdirac-togpu-00'
    outqueue = 'tcdirac-fromgpu-00'
    source_bucket = 'tcdirac-togpu-00'
    dest_bucket = 'tcdirac-fromgpu-00'
    world_comm = MPI.COMM_WORLD
    #seed()
    
    g = GPU(world_comm, inqueue, outqueue, source_bucket, dest_bucket)
    g.run()

                
                
        
