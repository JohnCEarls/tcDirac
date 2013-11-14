import time
import os, os.path
from multiprocessing import Queue,Process
import multiprocessing
import logging
import socket
import numpy as np
import boto
import boto.utils
from boto.s3.key import Key
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message
import random
import json

from gpu.datatransfer import RetrieverQueue, PosterQueue
import dtypes
import debug
from gpu.loader import LoaderBoss, LoaderQueue, MaxDepth
from gpu.results import PackerBoss, PackerQueue
from gpu import sharedprocesses
from gpu import data


class PosterProgress(Exception):
    pass

    

class Dirac:
    """
    Class for running dirac on the gpu.
    name: a unique identifier for this dirac instance
    directories: contains local storage locations
        directories['source'] : data source directorie
        directories['results'] : directory for writing processed data
        directories['log'] : directory for logging
    #these settings are retrieved from cluster master via the command queue
    s3: dict containing names of buckets for retrieving and depositing data
        s3['source'] : bucket for source data
        s3['results'] : bucket for writing results
    sqs: dict contain sqs queues for communication with data nodes and commands
        sqs['source'] : queue containing data files ready for download
        sqs['results'] : queue for putting result data name when processed
        sqs['command'] : queue containing instructions from master
        sqs['response'] : queue for communication with master
    """
    def __init__(self, directories, init_q ):
        self.name = self._generate_name()

        self.s3 = {'source':None, 'results':None} 
        self.sqs = {'source':None, 'results':None, 'command': self.name + '-command' , 'response': self.name + '-response' }
        self.directories = directories
        self._terminating = 0
        #terminating is state machine
        #zero means not terminating
        #one means soft kill on retriever, so hopefully the pipeline will runout
        #two means waiting for loader to clear queue
        #three means waiting for packer to pack all data
        #four means waiting for poster to send off data
        self._makedirs()
        self._get_settings( init_q )
        self._init_subprocesses()



    def run(self):
        try:
            while self._terminating < 5:
                res = self._main()
                if not res:
                    logging.info("%s: error in main <loader>" %s)
                self._heartbeat()
        except:
            #pop cuda context before we raise exception
            self._catch_cuda()
            logging.exception()
            raise


    def _main(self):
        """
        This runs the primary logic for gpu
        """
        try:
            db = self._loaderq.next_loader_boss()
        except MaxDepth:
            logging.info("%s: exceeded max depth for LoaderBoss" % name)
            return False
        except:
            return False
        
        db.clear_data_ready()
        expression_matrix = db.get_expression_matrix()
        gene_map = db.get_gene_map()
        sample_map = db.get_sample_map()
        network_map = db.get_network_map()
        exp = data.SharedExpression( expression_matrix )
        gm = data.SharedGeneMap( gene_map )
        sm = data.SharedSampleMap( sample_map )
        nm = data.SharedNetworkMap( network_map )
        srt,rt,rms =  sharedprocesses.runSharedDirac( exp, gm, sm, nm, self.sample_block_size, self.npairs_block_size, self.nets_block_size )
        db.release_loader_data()
        db.set_add_data()
        pb = pb_q.next_packer_boss()
        rms.fromGPU( pb.get_mem() )
        pb.set_meta( my_f['file_id'], ( rms.buffer_nnets, rms.buffer_nsamples ), dtype['rms'] )
        pb.release() 
        return True

    def _heartbeat(self):
        """
        Phones home to let master know the status of our gpu node
        """
        if self._hb >= self._hb_interval:
            self._hb = 0
            conn = boto.sqs.connect_to_region( 'us-east-1' )
            response_q = conn.create_queue( self.sqs['response'] )
            mess = self._generate_heartbeat_message()
            response_q.write( mess )
            if self._terminating > 0:
                self._terminator()
        else:
            self._hb += 1

    def _terminator(self):
        """
        #terminating is state machine
        #zero means not terminating
        #two means waiting for loader to clear queue
        #three means waiting for packer to pack all data
        #four means waiting for poster to send off data
        #five means we should exit main process
        """
        self._tcount += 1
        if self._tcount > 100:
            #we've tried to clean up too much
            logging.error("%s: Unable to exit cleanly, getting dirty" % self.name )
            for c in multiprocessing.active_children():
                c.terminate()

            raise Exception("Unable to exit cleanly")
        elif self._terminating == 1:
            #one means soft kill on retriever, so hopefully the pipeline will runout
            if not self._soft_kill_retriever():
                self._hard_kill_retriever()
            self._terminating = 2
        elif self._terminating == 2:
            try:
                db = self._loaderq.next_loader_boss()
            except MaxDepth:
                if self._source_q.empty():
                    self._loaderq.kill_all()
                    self._terminating = 3
                    self._hb_interval = 1
            except:
                #no loaders
                self._terminating = 3
                self._hb_interval = 1
        elif self._terminating == 3:
            ctr = 1
            while not self._packerq.no_data() and ctr < 10:
                time.sleep(1)
                ctr += 1
            self._packerq.kill_all()
            self._terminating = 4
        elif self._terminating == 4:
            ctr = 0
            while not self._results_q.empty() and ctr < 10:
                time.sleep(1)
                ctr += 1
            self._posterq.kill_all()
            self._posterq.clean_up()
            self._terminating = 5

    def _check_commands(self):
        conn = boto.sqs.connect_to_region( 'us-east-1' )
        command_q = conn.create_queue( self.sqs['command'] )
        for mess in command_q.get_messages(num_messages=10):
            self._handle_command(json.loads(mess.get_body()))

    def _handle_command( self, command):
        if command['message-type'] == 'termination-notice':
            self.__soft_kill_retriever()
            self._terminating = 1

    


    def _generate_heartbeat_message(self):
        message = {}
        message['message-type'] = 'gpu-heartbeat'
        message['name'] = self.name
        message['num-packer'] = self._packerq.num_sub()
        message['num-poster'] = self._posterq.num_sub()
        message['num-retriever'] = self._retrieverq.num_sub()
        message['num-loader'] = self._loaderq.num_sub()
        message['source-q'] = self._source_q.qsize()
        message['result-q'] = self._result_q.qsize()
        message['terminating'] = self._terminating
        message['time'] = time.time()
        return Message(body=json.dumps(message))
    

    def _init_gpu(self):
        cuda.init()
        dev = cuda.Device( self.gpu_id )
        self.ctx = dev.make_context()

    def _catch_cuda(self):
        try:
            self.ctx.pop()
        except:
            logging.exception("%s: unable to successfully clear context, this attempt most likely after a different error" %self.name)


    def _get_settings(self, init_q_name):
        """
        Alert master to existence
        Get settings
        """
        

        conn = boto.sqs.connect_to_region( 'us-east-1' )
        init_q = None
        ctr = 0
        self._generate_command_queues()
        while init_q is None and ctr < 6:
            
            init_q = conn.get_queue( init_q_name  )
            time.sleep(1+ctr**2)
            
            ctr += 1
        if init_q is None:
            logging.error("%s: Unable to connect to init q" %self.name)
            raise Exception("Unable to connect to init q")
        
        md =  boto.utils.get_instance_metadata()
        self._availabilityzone = md['placement']['availability-zone']
        self._region = self._availabilityzone[:-1]
        message = {'message-type':'gpu-init', 'name': self.name, 'instance-id': md['instance-id'], 'command' : self.sqs['command'], 'response' : self.sqs['response'], 'zone':self._availabilityzone }
        m = Message(body=json.dumps(message))
        init_q.write( m )
        command_q = conn.create_queue( self.sqs['command'] )
        command = None
        ctr = 0
        while command is None and ctr < 10 :
            command = command_q.read(  wait_time_seconds=20 ) 
            ctr += 1
        if command is None:
            self._delete_command_queues()
            logging.error("%s: Attempted to retrieve setup and no instructions received." % name)
            raise Exception("Waited 200 seconds and no instructions, exitting.")
        
        parsed = json.loads(command.get_body())
        self.sqs['results'] = parsed['result-sqs']
        self.sqs['source'] = parsed['source-sqs']
        self.s3['source'] = parsed['source-s3']
        self.s3['results'] = parsed['result-s3']
        self.data_settings = parsed['data-settings']
        self.gpu_id = parsed['gpu-id']
        self.sample_block_size = parsed['sample-block-size']
        self.pairs_block_size = parsed['pairs-block-size']
        self.nets_block_size = parsed['nets-block-size']
        self._hb_interval = parsed['heartbeat-interval']

 
        init_q.delete_message( command )
        logging.info("%s: sqs< %s > s3< %s > ds< %s > gpu_id< %s >" % (self.name, str(self.sqs), str(self.s3), str(self.data_settings), str(self.gpu_id)) )


    def _generate_command_queues(self):
        """
        Create the command queues for this process
        """
        conn = boto.sqs.connect_to_region( 'us-east-1' )
        response_q = conn.create_queue( self.sqs['response'] )
        command_q = conn.create_queue( self.sqs['command'] )
        #check
        command_q = None
        while command_q is None:
            command_q = conn.get_queue( self.sqs['command'] )
            time.sleep(1)
        response_q = None
        while response_q is None:
            response_q = conn.get_queue( self.sqs['response'] )
            time.sleep(1)

    def _delete_command_queues(self):
        conn = boto.sqs.connect_to_region( 'us-east-1' )
        command_q = conn.get_queue( self.sqs['command'] )
        command_q.delete()
        response_q = conn.get_queue( self.sqs['response'] )
        response_q.delete()

    def _generate_name(self):
        md =  boto.utils.get_instance_metadata()
        return md['instance-id'] + '_' + str(random.randint(10000,99999))

    def _init_subprocesses(self):
        logging.info("%s: Initializing subprocesses" % self.name)
        self._source_q = Queue()#queue containing names of source data files for processing
        self._result_q = Queue()#queue containing names of result data files from processing
        self._retrieverq = RetrieverQueue( self.name + "_rq", self.directories['source'], self._source_q, self.sqs['source'], self.s3['source'] )
        self._posterq = PosterQueue( self.name + "_poq", self.directories['results'], self._result_q, self.sqs['results'], self.s3['results'] )
        self._loaderq = LoaderQueue( self.name + "_lq", self._source_q, self.directories['source'], data_settings = self.data_settings['source'] )
        self._packerq = PackerQueue( self.name + "_paq", self._results_q, self.directories['results'], data_settings = self.data_settings['results'] )
        logging.info("%s: Subprocesses Initialized" % self.name )
        

    def start_subprocesses(self):
        logging.info("%s: starting subprocesses")
        self._retrieverq.add_retriever(5)
        self._posterq.add_poster(5)
        self._loaderq.add_loader_boss(5)
        self._packerq.add_packer_boss(5)

    def _soft_kill_retriever(self):
        logging.info("%s: attempting soft_kill_retriever")
        if not self._retriever_q.all_dead():
            self._retrieverq.kill_all()
            return False
        else:
            self._retrieverq.clean_up()
            return True

    def _hard_kill_retriever(self):
        self._retrieverq.clean_up()

    def _soft_kill_poster(self):
        """
        Waits until all files are sent and then gives kill signal to posterq
        If the result_q is not shrinking, raises PosterProgress exception
        """
        curr_qsize = 10000
        tries = 0
        while not self._result_q.empty():
            if self._result_q.qsize() < curr_qsize:
                curr_qsize =  self._result_q.qsize()
                time.sleep(2)
                tries += 1
            else:
                raise PosterProgress("Poster not making progress %i" % self._result_q.qsize())
        self._posterq.kill_all()
        self._posterq.clean_up()
        return True    

    def _hard_kill_poster(self):
        self._posterq.kill_all()
        self._posterq.clean_up()

                


    def _loadbalance(self):
        raise Exception("Unimplemented.")

    def _makedirs(self):
        for k, p in directories.iteritems():
            if not os.path.exists(p):
                try:
                    os.makedirs(p)
                except:
                    #might have multiple procs trying this, if already done, ignore
                    pass

def mockMaster( master_q = 'tcdirac-master'):
    conn = boto.sqs.connect_to_region( 'us-east-1' )
    in_q = conn.get_queue( master_q )
    m = None
    while m is None:
        logging.info("MM waiting for message.. ")
        m = in_q.read( wait_time_seconds=20 )
    in_q.delete_message(m)
    settings = json.loads(m.get_body())
    rq = conn.get_queue( settings['response'] )
    cq = conn.get_queue( settings['command'] )

    m = Message(body=get_gpu_message())

    cq.write(m)

def get_gpu_message():
    parsed = {}
    parsed['result-sqs'] = 'tcdirac-from-gpu-00'
    parsed['source-sqs'] = 'tcdirac-to-gpu-00'
    parsed['source-s3'] = 'tcdirac-togpu-00'
    parsed['result-s3'] = 'tcdirac-fromgpu-00'
    dsize = {'em':10000, 'gm':1000, 'sm':1000, 'nm':1000, 'rms':1000}
    dtype = {'em':np.float32, 'gm':np.int32, 'sm':np.int32,'nm':np.int32,'rms':np.float32 }
    ds = []
    for k in ['em', 'gm', 'sm', 'nm']:
        ds.append( (k, dsize[k], dtypes.to_index(dtype[k])))
    
    parsed['data-settings'] = {'source':ds}
    ds = [('rms', dsize['rms'], dtypes.to_index(dtype['rms'])) ]
    parsed['data-settings']['results'] = ds
    parsed['gpu-id'] = 0

    parsed['sample-block-size'] = 32
    parsed['pairs-block-size'] = 16
    parsed['nets-block-size'] = 8
    
    return json.dumps(parsed)

if __name__ == "__main__":
    debug.initLogging("tcdirac_gpu_test.log", logging.INFO, st_out=True)
    p = Process(target=mockMaster)
    p.start()

    directories = {}
    directories['source'] = '/scratch/sgeadmin/source' 
    directories['results'] = '/scratch/sgeadmin/results'
    directories['log'] = '/scratch/sgeadmin/log'

    init_q = 'tcdirac-master'

    d = Dirac( directories, init_q )
    time.sleep(10)
    d._delete_command_queues()
