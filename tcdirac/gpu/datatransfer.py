import sys

import inspect, os, os.path
if os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) == '/home/sgeadmin/hdproject/tcDirac/tcdirac/gpu':
    #if we are running this from dev dir, need to add tcdirac to the path
    sys.path.append('/home/sgeadmin/hdproject/tcDirac')

from multiprocessing import Process, Queue, Lock, Value, Event, Array
from Queue import Empty, Full

import boto
from boto.s3.key import Key
import boto.sqs
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message

import os
import os.path

import tcdirac
from tcdirac import static
import logging
import time
import json
import random

class Retriever(Process):
    def __init__(self, name, in_dir,  q_ret2gpu, evt_death, sqs_name, s3bucket_name, max_q_size):
        Process.__init__(self, name=name)
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.logger.debug( "Init: in_dir<%s> sqs_name<%s> s3bucket_name<%s> max_q_size<%i>", (in_dir, sqs_name, s3bucket_name, max_q_size) )
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
                self.logger.warning("starving")
                time.sleep(random.randint(1,10))

    def run_once(self):
        messages = self._sqs_q.get_messages(10)
        m_count = 0
        for message in messages:
            try:
                m = json.loads(message.get_body())
                for f in m['f_names']:
                    self.download_file( f )
                    self.logger.debug("Downloaded <%s>" % f)
                cont = True
                while cont:
                    try:
                        self.q_ret2gpu.put( m, timeout=10 )
                        cont = False
                    except Full:
                        self.logger.warning("queue_full" )
                        if self.evt_death.is_set():
                            return m_count
                self._sqs_q.delete_message(message)
                m_count += 1
            except:
                self.logger.exception("While trying to download files" )                
        return m_count
            
    def _connect_s3(self):
        conn = boto.connect_s3()        
        b = conn.get_bucket( self.s3bucket_name )
        return b

    def _connect_sqs(self):
        conn = boto.sqs.connect_to_region('us-east-1')
        q = conn.create_queue( self.sqs_name )
        return q

    def download_file(self, file_name):
        k = Key(self._s3_bucket)
        k.key = file_name
        k.get_contents_to_filename( os.path.join(self.in_dir, file_name) )


class RetrieverQueue:
    def __init__(self,  name, in_dir,  q_ret2gpu,sqs_name, s3bucket_name):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.logger.debug( "Init: in_dir<%s> sqs_name<%s> s3bucket_name<%s> max_q_size<%i>", (in_dir, sqs_name, s3bucket_name) )
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
            self._retrievers.append( Retriever(self.name + "_Retriever_" + str(num), self.in_dir,  self.q_ret2gpu, evt_death, self.sqs_name, self.s3bucket_name, max_q_size=10*(num+1)  ) )
            self._reaper.append(evt_death)
            self._retrievers[-1].daemon = True
            self._retrievers[-1].start()
            self.add_retriever(num - 1)

    def remove_retriever(self):
        if len(self._retrievers) == 0:
            raise Exception("Attempt to remove retriever from empty queue")
        self.logger.info("removing retriever")
        self._reaper[-1].set()
        ctr = 0
        while self._retrievers[-1].is_alive() and ctr < 10:
            time.sleep(.2)
            ctr += 1
        if self._retrievers[-1].is_alive():
            self._retrievers[-1].terminate()
        self._reaper = self._reaper[:-1]
        self._retrievers = self._retrievers[:-1]


    def repair(self):
        for i, d in enumerate(self._reaper):
            if d.is_set():
                p = self._retrievers[i]
                if p.is_alive():
                    p.terminate()
                p.join(.5)    
            d.clear()
            self._retrievers[i] =  Retriever(self.name + "_Retriever_" + str(i)+"_repaired", self.in_dir,  self.q_ret2gpu, d, self.sqs_name, self.s3bucket_name, max_q_size=10*i)

    def kill_all(self):
        for r in self._reaper:
            r.set()

    def all_dead(self):
        for r in self._retrievers:
            if r.is_alive():
                return False
        return True

    def clean_up(self):
        for r in self._retrievers:
            if r.is_alive():
                r.terminate()
        for r in self._retrievers:
            r.join()
        self._retrievers = []
        self._reaper = []

    def num_sub(self):
        count = 0
        for r in self._retrievers:
            if r.is_alive():
                count += 1
        return count



class Poster(Process):
    def __init__(self, name, out_dir,  q_gpu2s3, evt_death, sqs_name, s3bucket_name):
        Process.__init__(self, name=name)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.logger.debug( "Init: out_dir<%s> sqs_name<%s> s3bucket_name<%s>", (out_dir, sqs_name, s3bucket_name) )
        self.q_gpu2s3 = q_gpu2s3
        self.sqs_name = sqs_name
        self._sqs_q = self._connect_sqs()
        self.s3bucket_name = s3bucket_name
        self._s3_bucket = self._connect_s3()
        self.out_dir = out_dir
        self.evt_death = evt_death
        

    def run(self):
        self.logger.info("starting...")
        while not self.evt_death.is_set():
            self.run_once()

    def run_once(self):
        try:
            f_info = self.q_gpu2s3.get(True, 3)
            self.upload_file( f_info['f_name'] )
            m = Message(body= json.dumps(f_info) )
            self._sqs_q.write( m )
        except Empty:
            self.logger.info("starving")
            if self.evt_death.is_set():
                self.logger.info("Exiting...")
                return
        except:
            self.logger.exception("exception in run_once")
            self.evt_death.set()

    def _connect_s3(self):
        conn = boto.connect_s3()        
        b = conn.get_bucket( self.s3bucket_name )
        return b

    def _connect_sqs(self):
        conn = boto.sqs.connect_to_region('us-east-1')
        q = conn.create_queue( self.sqs_name )
        return q

    def upload_file(self, file_name):
        k = Key(self._s3_bucket)
        k.key = file_name
        k.set_contents_from_filename( os.path.join(self.out_dir, file_name), reduced_redundancy=True )
        self.logger.debug("Deleting <%s>" % (os.path.join(self.out_dir, file_name)))
        os.remove(os.path.join(self.out_dir, file_name))


class PosterQueue:
    def __init__(self,  name, out_dir,  q_gpu2s3,sqs_name, s3bucket_name):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.logger.debug( "Init: out_dir<%s> sqs_name<%s> s3bucket_name<%s>", (out_dir, sqs_name, s3bucket_name) )
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
            self._posters.append( Poster(self.name + "_Poster_" + str(num), self.out_dir,  self.q_gpu2s3, evt_death, self.sqs_name, self.s3bucket_name))
            self._reaper.append(evt_death)
            self._posters[-1].daemon = True
            self._posters[-1].start()
            self.add_poster(num - 1)
            

    def remove_poster(self):
        if len(self._posters) == 0:
            raise Exception("Attempt to remove poster from empty queue")
        self.logger.info("removing poster")
        self._reaper[-1].set()
        ctr = 0
        while self._posters[-1].is_alive() and ctr < 10:
            time.sleep(.2)
            ctr += 1
        if self._posters[-1].is_alive():
            self._posters[-1].terminate()
        self._reaper = self._reaper[:-1]
        self._posters = self._posters[:-1]




    def repair(self):
        for i, d in enumerate(self._reaper):
            if d.is_set():
                p = self._posters[i]
                if p.is_alive():
                    p.terminate()
                p.join(.5)    
            d.clear()
            self.logger.warning("Repairing poster<%i>" % i)
            self._posters[i] =  Poster(self.name + "_p" + str(i)+"_repaired", self.out_dir,  self.q_gpu2s3, d, self.sqs_name, self.s3bucket_name)
            

    def _sp_alive(self):
        for d in self._reaper:
            if d.is_set():
                return False
        return False

    def kill_all(self):
        self.logger.info("sending death signals")
        for r in self._reaper:
            r.set()


    def clean_up(self):
        for r in self._posters:
            if r.is_alive():
                time.sleep(2)
                r.terminate()
        for r in self._posters:
            r.join()
        self._posters = []
        self._reaper = []
        self.logger.info("Complete...")



    def num_sub(self):
        count = 0
        for r in self._posters:
            if r.is_alive():
                count += 1
        return count


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
    for i in my_files.itervalues():
        print i
        ctr += 1
        if ctr > 10:
            break

    pq_sqs = 'tcdirac-fromgpu-00'
    rq_sqs = 'tcdirac-togpu-00'    
    pq_bucket = 'tcdirac-togpu-00'
    rq_bucket = 'tcdirac-togpu-00'

    q_gpu2s3 = Queue()
    out_dir = s_dir
    name="PQ"
    pq = PosterQueue(  name, out_dir,  q_gpu2s3, pq_sqs, pq_bucket)
    pq.add_poster(10)
    for i in my_files.itervalues():
        if len(i['f_names']) == 4:
            for f_name in i['f_names']:
                q_gpu2s3.put({'file_id':i['file_id'], 'f_name': f_name})
            ctr += 1
            conn = boto.connect_sqs()
            q = conn.create_queue( rq_sqs  )
            
            print "shitpoker"
            print json.dumps( i )
            m = Message(body=json.dumps( i ))
            print m.get_body()
            q.write( m )
            if ctr > 10:
                break

    while not q_gpu2s3.empty(): 
        time.sleep(1)

    time.sleep(10)
    pq.kill_all()
    time.sleep(2)
    pq.clean_up()

    q_s32gpu = Queue()
    name = "RQ"

    rq = RetrieverQueue( name, out_dir,q_s32gpu ,rq_sqs, pq_bucket)
    rq.add_retriever(10)

    time.sleep(3)
    
    while q_s32gpu.qsize() < 100:
        print "z"
        time.sleep(1)
    rq.kill_all()
    time.sleep(2)

    rq.clean_up()
    while not q_s32gpu.empty():
        print q_s32gpu.get()
    
    
    

