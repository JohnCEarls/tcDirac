import sys

import inspect, os, os.path
if os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) == '/home/sgeadmin/hdproject/tcDirac/tcdirac/gpu':
    #if we are running this from dev dir, need to add tcdirac to the path
    sys.path.append('/home/sgeadmin/hdproject/tcDirac')

from multiprocessing import Process, Queue, Lock, Value, Event, Array
from Queue import Empty, Full

import boto
from boto.s3.key import Key
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message

import os
import os.path

import tcdirac
import logging
import time
class Retriever(Process):
    def __init__(self, name, in_dir,  q_ret2gpu, evt_death, sqs_name, s3bucket_name, max_q_size):
        Process.__init__(self, name=name)
        self.q_ret2gpu = q_ret2gpu
        self.sqs_name = sqs_name
        self._sqs_q = self._connect_sqs()
        self.s3bucket_name = s3bucket_name
        self._s3_bucket = self._connect_s3()
        self.in_dir = in_dir
        self.evt_death = evt_death
        self.max_q_size = max_q_size
        

    def run(self):
        while not self.evt_death.is_set():
            if self.q_ret2gpu.qsize() < self.max_q_size:
                messages = self.run_once()
            if messages < 10 and not self.evt_death.is_set():
                logging.debug("%s: zzzzzzz" % self.name )
                time.sleep(random.randint(1,10))


    def run_once(self):
        messages = self._sqs_q.get_messages(10)
        m_count = 0
        for message in messages:
            try:
                m = json.loads(message.get_body())
                for f in m['f_names']:
                    self.download_file( f )
                cont = True
                while cont:
                    try:
                        self.q_ret2gpu.put( m, timeout=10 )
                        cont = False
                    except Full:
                        logging.debug("%s: queue_full" % self.name )
                        if self.evt_death.is_set():
                            return m_count
                self._sqs_q.delete(m)
                m_count += 1
            except:
                logging.exception("%s: while trying to download files" % self.name)                
        return m_count
            

    def _connect_s3(self):
        conn = boto.connect_s3()        
        b = conn.get_bucket( self.s3bucket_name )
        return b

    def _connect_sqs(self):
        conn = boto.connect_sqs()
        q = conn.create_queue( self.sqs_name )
        return q

    def download_file(self, file_name):
        k = Key(self._s3_bucket)
        k.key = file_name
        k.get_contents_to_filename( os.path.join(self.in_dir, file_name) )


class RetrieverQueue:
    def __init__(self,  name, in_dir,  q_ret2gpu,sqs_name, s3bucket_name):
        self.name = name
        self.in_dir = in_dir
        self.q_ret2gpu= q_ret2gpu
        self.sqs_name = sqs_name
        self.s3bucket_name = s3bucket_name
        self._retrievers = []
        self._reaper = []

    def add_retriever(self, num=1):
        if num <= 0:
            return
        else:
            evt_death = Event()
            evt_death.clear()
            self._retrievers.append( Retriever(self.name + "_r" + str(num), self.in_dir,  self.q_ret2gpu, evt_death, self.sqs_name, self.s3bucket_name, max_q_size=10*num))
            self._reaper.append(evt_death)
            self._retrievers[-1].start()
            self.add_retriever(num - 1)

    def kill_all(self):
        for r in self._reaper:
            r.set()
    def clean_up(self):
        for r in self._retrievers:
            if r.is_alive():
                r.terminate()
        for r in self._retrievers:
            r.join()

class Poster(Process):
    def __init__(self, name, out_dir,  q_gpu2s3, evt_death, sqs_name, s3bucket_name):
        Process.__init__(self, name=name)
        self.q_gpu2s3 = q_gpu2s3
        self.sqs_name = sqs_name
        self._sqs_q = self._connect_sqs()
        self.s3bucket_name = s3bucket_name
        self._s3_bucket = self._connect_s3()
        self.out_dir = out_dir
        self.evt_death = evt_death
        

    def run(self):
        logging.info("%s: starting..." % self.name)
        while not self.evt_death.is_set():
            self.run_once()

    def run_once(self):
        try:
            f_info = self.q_gpu2s3.get(True, 3)
            self.upload_file( f_info['f_name'] )
            m = Message( json.dumps(f_info) )
            self._sqs_q.put( m )
        except Empty:
            if self.evt_death.is_set():
                logging.info("%s: exiting..."%self.name)
                return
            
            
            

    def _connect_s3(self):
        conn = boto.connect_s3()        
        b = conn.get_bucket( self.s3bucket_name )
        return b

    def _connect_sqs(self):
        conn = boto.connect_sqs()
        q = conn.create_queue( self.sqs_name )
        return q

    def upload_file(self, file_name):
        k = Key(self._s3_bucket)
        k.key = file_name
        k.set_contents_from_filename( os.path.join(self.out_dir, file_name), reduced_redundancy=True )

        os.remove(os.path.join(self.out_dir, file_name))


class PosterQueue:
    def __init__(self,  name, out_dir,  q_gpu2s3,sqs_name, s3bucket_name):
        self.name = name
        self.out_dir = out_dir
        self.q_gpu2s3= q_gpu2s3
        self.sqs_name = sqs_name
        self.s3bucket_name = s3bucket_name
        self._posters = []
        self._reaper = []

    def add_poster(self, num=1):
        if num <= 0:
            return
        else:
            evt_death = Event()
            evt_death.clear()
            self._posters.append( Poster(self.name + "_p" + str(num), self.out_dir,  self.q_gpu2s3, evt_death, self.sqs_name, self.s3bucket_name))
            self._reaper.append(evt_death)
            self._posters[-1].start()
            self.add_poster(num - 1)

    def kill_all(self):
        logging.info("%s: sending death signals"%self.name)
        for r in self._reaper:
            r.set()


    def clean_up(self):
        for r in self._posters:
            if r.is_alive():
                r.terminate()
        for r in self._posters:
            r.join()
        logging.info("%s: complete..."%self.name)



if __name__ == "__main__":
    import os
    import tcdirac.debug
    tcdirac.debug.initLogging("tcdirac_gpu_mptest.log", logging.INFO, st_out=True)

    s_dir = "/scratch/sgeadmin/"
    files = os.listdir(s_dir)
    my_files = {}
    
    for f in files:
        try:
            typ, fid, hsh = f.split('_')
            if fid not in my_files:
                my_files[fid] = {'file_id':fid, 'f_names':[]}
            if typ in ['em','gm','sm','nm']:
                my_files[fid]['f_names'].append(f)
            
        except:pass
               
    ctr = 0
    for i in my_files.iteritems():
        print i
        ctr += 1
        if ctr > 10:
            break

    pq_sqs = 'tcdirac-test00'
    bucket = 'tcdirac-togpu-00'

    q_gpu2s3 = Queue()
    out_dir = s_dir
    name="PQ"

    pq = PosterQueue(  name, out_dir,  q_gpu2s3, pq_sqs, bucket)
    pq.add_poster(5)

    time.sleep(10)
    pq.kill_all()
    time.sleep(4)
    pq.clean_up()



