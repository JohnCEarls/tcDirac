
import sys

import inspect, os, os.path
if os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) == '/home/sgeadmin/hdproject/tcDirac/tcdirac/gpu':
    #if we are running this from dev dir, need to add tcdirac to the path
    sys.path.append('/home/sgeadmin/hdproject/tcDirac')

import errno
import time
import logging
import itertools
from tempfile import TemporaryFile
import hashlib
import random
import cPickle

from multiprocessing import Process, Queue, Lock, Value, Event, Array
from Queue import Empty

import ctypes
import numpy as np
import scipy.misc
import pandas
from tcdirac import static
import pycuda.driver as cuda

import tcdirac.dtypes as dtypes
from tcdirac.gpu import processes

#from results import PackerBoss


class MaxDepth(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)



def kill_all(name, loaders, _ld_die_evt, loader_dist, file_q):
        """
        Kills all subprocesses
            called as subprocess by LoaderBoss in kill_all
        """
        logger = logging.getLogger(name)
        logger.debug("terminator: Killing subprocesses")
        temp_l = None
        for l in loaders.itervalues():
            l.die()
            temp_l = l
        _ld_die_evt.set()
        dead = False
        count = 1
        while not dead:
            time.sleep(1)
            dead = True 
            if _ld_die_evt.is_set():
                dead = False
            for l in loaders.itervalues():
                if l.events['die'].is_set():
                    dead = False
            if count >= 10:
                logger.error("Terminator: Unable to clear queues")
                return
        #put back unused data
        while not temp_l.q.empty():
            temp_d = {}
            t_check = []
            for k, a_loader in loaders.iteritems():
                if not a_loader.q.empty():
                    fname = a_loader.q.get()
                    part, rn, a_hash = fname.split('_')
                    t_check.append(rn)
                    temp_d[k] = fname
            failed = False
        
            for r in t_check[1:]:
                if r!=t_check[0]:
                    #order got screwed up. lost data
                    failed = True
            
            
            if not failed and len(t_check) == 4:
                logger.debug( "terminator: recycling")
                file_q.put( temp_d )
            else:
                logger.error( "terminator: data out of order [%s]" % (','.join(temp_d.itervalues()),))



class LoaderQueue:
    """
    Object containing a list of LoaderBosses for the gpu to pull from
    """
    def __init__(self,name, file_q, in_dir, data_settings):
        self.name=name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.logger.info("Initializing: dir<%s> data_settings<%s>" % (in_dir, str(data_settings)) )
        self.file_q = file_q #queue containing datasources
        self.in_dir = in_dir
        self.data_settings = data_settings
        self._bosses = []
        self._bosses_skip = []
        self._curr = -1

    def add_loader_boss(self, num=1):
        if num <= 0:
            return
        else:
            self._bosses.append( LoaderBoss('_'.join([self.name, 'LoaderBoss',str(len(self._bosses))]), self.file_q,self.in_dir, self.data_settings) )
            self._bosses_skip.append(0)
            self._bosses[-1].start()
            self._bosses[-1].set_add_data()
            self.add_loader_boss( num - 1)

    def next_loader_boss(self, time_out=0.1, max_depth=None):
        if len(self._bosses) == 0:
            raise Exception("No Loaders")
        if max_depth is None:
            max_depth = 2*len(self._bosses)#default max_depth to 2 passes of the queue
            logging.debug( "Max depth set[%i]", max_depth) 
        if max_depth <= 0:
            self.logger.debug("Exceeded Max Depth")
            raise MaxDepth("Max Depth exceeded")            

        self._curr = (self._curr + 1)%len(self._bosses)
        if self._bosses[self._curr].wait_data_ready(time_out=time_out):
            self._bosses_skip[self._curr] = 0
            return self._bosses[self._curr]
        else:
            self._bosses_skip[self._curr] += 1
            return self.next_loader_boss(time_out, max_depth=max_depth-1)

    def checkSkip(self, max_skip=3):
        over_limit = [i for i,l in enumerate(self._bosses_skip) if l > max_skip]
        temp = []
        for i in over_limit:
            temp.append(self._bosses[i])
            self._bosses[i].kill_all()
        for i in over_limit:
            self._bosses[i] =  LoaderBoss('_'.join([self.name,'LoaderBoss',str(i)]) , self.file_q, self.data_settings) 
            self._bosses_skip[i] = 0
        for l in temp:
            l.clean_up()

    def remove_loader_boss(self):
        if len(self._bosses) <=0:
            raise Exception("Attempt to remove Loader from empty LoaderQueue")
        temp = self._bosses[-1]
        temp.kill_all()
        self._bosses = self._bosses[:-1]
        self._bosses_skip = self._bosses_skip[:-1]
        self._curr = self._curr%len(self._bosses)
        temp.clean_up()

    def no_data(self):
        return self.file_q.empty()

    def kill_all(self):
        for l in self._bosses:
            l.kill_all()
            self.logger.info("killing [%s]" % l.name)
        for l in self._bosses:
            l.clean_up()
          
        self._bosses = []
        self._bosses_skip = []
        self._curr = -1 
    
    def set_data_settings(self, data_settings):
        self.data_settings = data_settings

   
    def num_sub(self):
        count = 0
        for r in self._bosses:
            if r.is_alive():
                count += 1
        return count
 


class LoaderBoss:
    """
    Object for initializing and interacting with the data loading modules
    name - a name for this set of loaders
    file_q - a queue for passing file names to the loaders.
    in_dir - the directory holding the data to be loaded
    data_settings - a list of tuples of the form (name, buffer size, data type)
        for example, [('exp', 200*20000, np.float32), ('sm',5*200, np.uint32),...]
        the names expected should be 
            'em' - expression matrix, 
            'gm' - gene map,
            'sm' - sample map,
            'nm' - network map
    """
    def __init__(self, name, file_q,in_dir,data_settings):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.file_q = file_q
        self.id_q = Queue()
        self.in_dir = in_dir
        self.data_settings = data_settings
        self.loaders = self._create_loaders('_'.join(['proc',name]), data_settings)
        self.loader_dist = self._create_loader_dist()
        self._terminator = Process( target=kill_all, args=(self.name + '-kill_all-sp', self.loaders, self._ld_die_evt, self.loader_dist, file_q))

    def get_file_id(self):
        """
        returns the file id for the currently processed data
            throws Queue.Empty, that means we are out of sync
        """
        return self.id_q.get(True, 3) 
 
    def get_expression_matrix(self):
        return self._get_loader_data('em')

    def get_gene_map(self):
        return self._get_loader_data('gm')

    def get_sample_map(self):
        return self._get_loader_data('sm')

    def get_network_map(self):
        return self._get_loader_data('nm')

    def release_expression_matrix(self):
        return self._release_loader_data('em')

    def release_gene_map(self):
        return self._release_loader_data('gm')

    def release_sample_map(self):
        return self._release_loader_data('sm')

    def release_network_map(self):
        return self._release_loader_data('nm')

    def release_loader_data(self):
        for l in self.loaders.keys():
            self._release_loader_data(l)

    def empty(self):
        """
        Returns true if no new data and all present data has been used
        """
        a_loader=self.loaders[self.data_settings[0][0]]
        self.logger.debug("Empty <fq:%s >"%( str(self.file_q.empty())))
        for k, a_loader in self.loaders.iteritems():
            self.logger.debug("empty loader <%s>" % ( k))
            self.logger.debug("Empty <loader q:%s >"%( str(a_loader.q.empty())))
            self.logger.debug("Empty <ad:%s >"%( str(a_loader.events['add_data'].is_set())))
            self.logger.debug("Empty <not dr:%s >"%( str(not a_loader.events['data_ready'].is_set())))
        return self.file_q.empty() and a_loader.q.empty() and a_loader.events['add_data'].is_set() and not a_loader.events['data_ready'].is_set()

    def start(self):
        """
        Starts the worker subprocesses
        """
        for l in self.loaders.itervalues():
            l.start()
        self.loader_dist.start()

    def kill_all(self):
        """
        Kills all subprocesses
        """
        self._terminator.start()

    def set_add_data(self, key=None):
        """
        Tells loaders to add next data
        """
        if key is None:
            for v in self.loaders.itervalues():
                v.events['add_data'].set()
        else:
            self.loaders[key].events['add_data'].set()

    def clear_data_ready(self):
        """
        Clears data ready, this means the data in shared memory is either being written to
        or has been read from.
        """
        for v in self.loaders.itervalues():
            v.events['data_ready'].clear()

    def wait_data_ready(self, time_out=3):
        """
        Waits for all loaders to have data ready.
        If any times out, returns False
        Otherwise True
        """
        ready = True
        for v in self.loaders.itervalues():
            if not v.events['data_ready'].wait(time_out):
                ready = False
        return ready

    def processes_running(self):
        """
        Returns True if all subprocesses are alive
        """
        pr = True
        for v in self.loaders.itervalues():
            if not v.process.is_alive():
                pr = False
        if not self.loader_dist.is_alive():
            pr = False
        return pr

    def _release_loader_data(self, key):
        return self.loaders[key].release_data()

    def _get_loader_data(self, key):
        return self.loaders[key].get_data()

    def _create_loader_dist(self):
        self._ld_die_evt = Event()
        return LoaderDist( '_'.join([self.name,'loader_dist']), self.file_q, self.id_q, self.loaders, self._ld_die_evt)

    def _create_loaders(self, name, data_settings):
        loaders = {}
        for name, dsize, dtype in data_settings:
            loaders[name] = self._create_loader( '_'.join([self.name,name]), dsize, dtype )
        return loaders

    def _create_loader(self,name, dsize, dtype):
        smem = self._create_shared(dsize, dtype)
        evts = self._create_events()
        file_q = Queue()
        l = Loader(file_q, evts, smem, self.in_dir, name+'-Loader')
        ls = LoaderStruct(name+'-loaderStruct',smem,evts,file_q, process=l)
        return ls

    def _create_shared(self, dsize, dtype):
        shared_mem = {}
        shared_mem['data'] = Array(dtypes.to_ctypes(dtype),dsize )
        shared_mem['shape'] = Array(dtypes.to_ctypes(np.int64), 2)
        for i in xrange(len(shared_mem['shape'])):
            shared_mem['shape'][i] = 0
        shared_mem['dtype'] = Value('i',dtypes.nd_dict[np.dtype(dtype)])
        return shared_mem

    def _create_events(self):
        events = {}
        events['add_data'] = Event()
        events['data_ready'] = Event()
        events['die'] = Event()
        return events

    def clean_up(self):
        self.logger.debug("Cleaning up subprocesses")
        temp_l = None
        for l in self.loaders.itervalues():
            if l.process.is_alive():
                l.process.terminate()
            l.process.join()
            temp_l = l
        self.logger.debug("loaders joined")
      
        if self.loader_dist.is_alive():
            self.loader_dist.terminate()
        self.loader_dist.join()
        self.logger.debug("loader_dist joined")
        if self._terminator.is_alive():
            self._terminator.terminate()
        self._terminator.join()

    def is_alive(self):
        for l in self.loaders.itervalues():
            if not l.process.is_alive():
                return False
        return True

class LoaderStruct:
    """
    Simple ds for interacting with loader subprocesses
    """
    def __init__(self,name,shared_mem,events,file_q,process=None):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.shared_mem = shared_mem
        self.events = events
        self.q = file_q
        self.process = process

    def start(self):
        """
        Starts subprocess
        """
        for e in self.events.itervalues():
            e.clear()
        self.process.start()

    def die(self):
        """
        informs subprocess it is time to die
        """
        self.events['die'].set()

    def get_data(self):
        """
        Returns the np array wrapping the shared memory
        Note: when done with the data, you must call release_data()
            a lock on the shared memory is acquired
        """
        shared_mem = self.shared_mem
        for m in shared_mem.itervalues():
            l = m.get_lock()
            l.acquire()
        myshape = np.frombuffer(shared_mem['shape'].get_obj(),dtype=int)
        t_shape = []
        N = 1
        for i in myshape:
            if i > 0:
                t_shape.append(i)
                N = N * i
        t_shape = tuple(t_shape)
        #Note: this is not a copy, it is a view
        #test with np.may_share_memory or data.ctypes.data
        data =  np.frombuffer(shared_mem['data'].get_obj(), dtype=dtypes.nd_list[shared_mem['dtype'].value])
        data = data[:N]
        data = data.reshape(t_shape)
        return data

    def release_data(self):
        """
        Releases lock on shared memory
            throws assertion error if lock not held by process
        """
        shared_mem = self.shared_mem
        for m in shared_mem.itervalues():
            l = m.get_lock()
            l.release()


class LoaderDist(Process):
    """
    takes in data from a single q and distributes it to loaders
    """
    def __init__(self, name, in_q, out_q, loaders, evt_death):
        Process.__init__(self, name=name)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.in_q = in_q
        self.out_q = out_q
        self.loaders = loaders
        self.proto_q = loaders[loaders.keys()[0]].q
        self.evt_death= evt_death

    def run(self):
        self.logger.debug("%s: starting..."% self.name)
        while not self.evt_death.is_set():
            try:
                if self.proto_q.qsize() < 2 + random.randint(2,10):
                    f_dict = self.in_q.get(True, 2)
                    self.logger.debug("%s: distributing <%s>" % ( self.name, f_dict['file_id']) )
                    self.out_q.put(f_dict)
                    for k,v in self.loaders.iteritems():
                        v.q.put(f_dict[k])
                else:
                    self.logger.debug("%s: sleeping due to full q"%self.name)
                    time.sleep(1)
            except Empty:#thrown by in_#thrown by in_qq
                self.logger.debug("%s: starving..."%self.name)
                pass
        self.evt_death.clear()
        self.logger.info("%s: exiting..." % (self.name,))
        
    
class Loader(Process):
    def __init__(self, inst_q, events, shared_mem, in_dir, name, add_data_timeout=10, inst_q_timeout=3):
        """
            inst_q = mp queue that tells process next file name
            events = dict of mp events
                events['add_data'] mp event, true means add data 
                events['data_ready'] mp event, true means data is ready to be consumed
                events['die'] event with instructions to die
            shared_mem = dict containing shared memory
                shared_mem['data'] = shared memory for numpy array buffer
                shared_mem['shape'] = shared memory for np array shape
                shared_mem['dtype'] = shared memory for np array dtype
            in_dir = string encoding location where incoming np data is being written and should be read from
            name = process name(up to you to make it unique)
            add_data_timeout = time in seconds before you check for death when waiting for gpu to release memory
            inst_q_timeout = time in seconds before you check for death when waiting for new filename
        """
        Process.__init__(self, name=name)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.daemon = True 
        self.instruction_queue = inst_q
        self.smem_data = shared_mem['data']
        self.smem_shape = shared_mem['shape']
        self.smem_dtype = shared_mem['dtype']
        self.evt_add_data = events['add_data']
        self.evt_data_ready = events['data_ready']
        self.evt_die = events['die']
        self.in_dir = in_dir
        self._ad_timeout = add_data_timeout
        self._iq_timeout = inst_q_timeout


    def _load_mem(self, np_array):
        """
        loads data from a numpy array into the provided shared memory
        """
        shape = np_array.shape
        dt_id = dtypes.nd_dict[np_array.dtype]
        size = np_array.size
        np_array = np_array.reshape(size)

        self.logger.debug("writing to shared memory")  
        with self.smem_data.get_lock():
            self.smem_data[:size] = np_array[:]
        with self.smem_shape.get_lock():
            self.smem_shape[:len(shape)] = shape[:]
        with self.smem_dtype.get_lock():
            self.smem_dtype.value = dt_id
        self.logger.debug("shared memory copy complete" )  


    def _clear_mem(self):

        with self.smem_data.get_lock():
            temp = np.frombuffer(self.smem_data.get_obj(), np.float32)
            temp.fill(0)


    def _get_data(self, fname):
        return np.load(os.path.join(self.in_dir, fname))

    def run(self):  
        self.logger.info("Starting ") 
        old_md5 = '0'
        #get file name for import
        fname = self.instruction_queue.get()
        new_md5 = self._get_md5(fname)
        self.logger.debug("loading file <%s>" %( fname))
        data = self._get_data(fname)
        self.logger.debug("<%s> loaded %f MB " % ( fname, data.nbytes/1048576.0))
        while True:
            self.evt_add_data.wait(self._ad_timeout)
            if self.evt_add_data.is_set():
                self.logger.debug(" loading data into mem ") 
                self._load_mem( data ) 
                self.logger.debug(" clearing evt_add_data" )
                self.evt_add_data.clear()
                self.logger.debug(" setting evt_data ready")
                self.evt_data_ready.set()
                self.logger.debug(" getting new file ") 
                fname = None
                while fname is None:
                    try:
                        fname = self.instruction_queue.get(True, self._iq_timeout) 
                    except Empty:
                        self.logger.debug(" fname timed out ")
                        if self.evt_die.is_set():
                            self.evt_die.clear()
                            self.logger.info(" exiting... " )  
                            return
                self.logger.debug(" new file <%s>" %(fname))
                old_md5 = new_md5
                new_md5 = self._get_md5(fname)
                if new_md5 != old_md5:
                    #self._clear_mem()
                    data = self._get_data(fname)
                    self.logger.debug(" <%s> loaded %f MB " % (fname, data.nbytes/1048576.0))
                else:
                    self.logger.debug(" same data, recycle reduce reuse" )
            elif self.evt_die.is_set():
                self.evt_die.clear()
                self.logger.info("Exiting... " )  
                return

    def _get_md5(self, fname):
        """
        Given a formatted filename, returns the precalculated md5 (really any kind of tag)
        """
        return fname.split('_')[-1]

            
def testLoader(pid=0):
    bdir = '/scratch/sgeadmin/'
    odir = '/scratch/sgeadmin/'
    np_list = []
    inst_q = Queue()
    results_q = Queue()
    check_cp = False #check whether in == out, if False, comparing speed
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
    dsize = {'em':0, 'gm':0, 'sm':0, 'nm':0}
    dtype = {'em':np.float32, 'gm':np.uint32, 'sm':np.uint32,'nm':np.uint32}

    check_list = []

    for i in range(10):
        f_dict = {}
        fake = genFakeData( 200, 20000) 
        for i in range(2):
            f_dict['file_id'] = str(random.randint(1000,10000))
            for k,v in fake.iteritems():
                f_hash = hashlib.sha1(v).hexdigest()
                f_name = '_'.join([ k, f_dict['file_id'], f_hash])
                with open(os.path.join( bdir, f_name),'wb') as f:
                    np.save(f, v)
                if v.size > dsize[k]:
                    dsize[k] = v.size
                f_dict[k] = f_name
            inst_q.put( f_dict )
            check_list.append(f_dict)
    data_settings = []
    for k,b in dsize.iteritems():   
        data_settings.append((k, b, dtype[k]))
 
    db = LoaderBoss(str(pid),inst_q,bdir,data_settings)
    pb = PackerBoss(str(pid), results_q, odir, (dsize['em'], dtype['em']) )
    db.start()
    pb.start()
    db.set_add_data()
    for f_dict in check_list:
        while not db.wait_data_ready(time_out=4):
            print "chillin"
            pass
        db.clear_data_ready()
        
        for k,v in f_dict.iteritems():
            if k != 'file_id':
                sdata = db._get_loader_data(k)
                if check_cp:
                    compare_data = np.load(os.path.join( bdir, v ))
                   
                    logging.info( "Tester: Copy Matches %s" % (str(np.allclose(sdata, compare_data)),))
                db._release_loader_data(k)
                db.set_add_data(k)
                logging.debug("tester: set add data<%s>" % k)
                
            elif False:#leaving this to allow testing of Packer Boss
                #should prob separate tests
                assert db.get_file_id() == v, "Out of order"
    if not db.loaders['em'].events['add_data'].is_set():
        raise Exception("Fuck you Jobu")
    while not db.empty():
        print "not empty"
        time.sleep(.5)
    for f_dict in check_list:
        while True:
            if pb.ready():
                print "Tester: packer running"
                compare_data = np.load(os.path.join( bdir, f_dict['em'] ))
                cdata = compare_data.flatten()
                sm = pb.get_mem()
                sm[:len(cdata)] = cdata[:]
                b_shape = compare_data.shape
                true_shape =  tuple( map(lambda x: x/2, compare_data.shape) )
                pb.set_meta(db.get_file_id(), b_shape, cdata.dtype, true_shape )
                pb.release()
                break
            else:
                print "Tester: packer not ready"
                time.sleep(.1)

    self.logger.info( "Tester: no data, all processed, killing sp")
    db.kill_all()
    time.sleep(10)
    db.clean_up()
    pb.kill()
    pb.clean_up()
    while not results_q.empty():
        print results_q.get()
 
    logging.info( "Tester: exitted gracefully")
    

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
    gene_map = np.array(gm_idx).astype(np.uint32)
    #print gene_map[:20]
    sample_map = np.random.randint(low=0,high=len(samples), size=(len(samples),neighbors))
    sample_map = sample_map.astype(np.uint32)
    #print sample_map[:,:3]
    network_map = np.array(net_map).astype(np.uint32)

    data = {}
    data['em'] = expression_matrix
    data['gm'] = gene_map
    data['sm'] = sample_map
    data['nm'] = network_map
    return data

if __name__ == "__main__":
    import tcdirac.debug
    tcdirac.debug.initLogging("tcdirac_gpu_mptest.log", logging.DEBUG, st_out=True)
    testLoader(1)
    """
    temp = []
    for i in range(1):
        p = Process(target=testLoader, args=(i,))
        temp.append(p)
        p.start()
    for p in temp:
        p.join()"""
