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

import boto
from boto.s3.key import Key
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message

import ctypes
import numpy as np
import scipy.misc
import pandas

import pycuda.driver as cuda

import tcdirac.dtypes as dtypes
from tcdirac.gpu import processes
from loader import LoaderBoss, LoaderQueue, MaxDepth
from results import PackerBoss, PackerQueue
import sharedprocesses
import data

def testAccuracy(pid=0):
    in_dir = '/scratch/sgeadmin/'
    odir = '/scratch/sgeadmin/'
    np_list = []
    sample_block_size = 32
    npairs_block_size = 16
    nets_block_size = 8

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
    dsize = {'em':0, 'gm':0, 'sm':0, 'nm':0, 'rms':0}
    dtype = {'em':np.float32, 'gm':np.int32, 'sm':np.int32,'nm':np.int32,'rms':np.float32 }

    check_list = []
    cuda.init()
    unique_fid = set()
    dev = cuda.Device(0)
    ctx = dev.make_context()
    for i in range(10):
        fake = genFakeData( 200, 20000)
        p_hash = None
        for i in range(2):
            f_dict = {}
            f_id = str(random.randint(10000,100000))
            while f_id in unique_fid:
                f_id = str(random.randint(10000,100000))
            f_dict['file_id'] = f_id
            unique_fid.add(f_id)

            for k,v in fake.iteritems():
                if k == 'em':
                    exp = data.Expression(v)
                    exp.createBuffer(sample_block_size, buff_dtype=np.float32)
                    v = exp.buffer_data
                    t_nsamp = exp.orig_nsamples
                elif k == 'sm':
                    sm = data.SampleMap(v)
                    sm.createBuffer(sample_block_size, buff_dtype=np.int32)
                    v = sm.buffer_data
                elif k == 'gm':
                    gm = data.GeneMap(v) 
                    gm.createBuffer( npairs_block_size, buff_dtype=np.int32)
                    v = gm.buffer_data
                elif k == 'nm':
                    nm = data.NetworkMap(v)
                    nm.createBuffer( nets_block_size, buff_dtype=np.int32 )
                    v = nm.buffer_data

                f_hash = hashlib.sha1(v).hexdigest()
                if k == 'em':
                    if p_hash is None:
                        p_hash = f_hash
                        p_temp = v.copy()
                    else:
                        assert p_hash == f_hash, str(v) + " " + str(p_temp)
                        p_hash = None
                        p_temp = None
                f_name = '_'.join([ k, f_dict['file_id'], f_hash])
                with open(os.path.join( in_dir, f_name),'wb') as f:
                    np.save(f, v)
                if v.size > dsize[k]:
                    dsize[k] = v.size
                f_dict[k] = f_name
            srt,rt,rms = processes.runDirac(exp.orig_data, gm.orig_data, sm.orig_data,nm.orig_data, sample_block_size, npairs_block_size, nets_block_size, True)
            """
            uncomment to compare srt and rts
            srt.fromGPU()
            np.save(os.path.join(in_dir, 'srt_'+ f_dict['file_id'] + '_single'), srt.res_data)
            rt.fromGPU()
            np.save(os.path.join(in_dir, 'rt_'+ f_dict['file_id'] + '_single' ), rt.res_data)
            """
            rms.fromGPU(res_dtype=np.float32)
            np.save(os.path.join(in_dir, 'rms_'+ f_dict['file_id'] + '_single'), rms.res_data)

            rms = data.RankMatchingScores( nm.buffer_nnets, sm.buffer_nsamples )
            
            rms.createBuffer(  sample_block_size, nets_block_size, buff_dtype=np.float32)
            if rms.buffer_data.size > dsize['rms']:
               dsize['rms'] = rms.buffer_data.size 
            inst_q.put( f_dict )
            check_list.append(f_dict)
    data_settings = []
    for k,b in dsize.iteritems():
        if k in ['em','sm','gm','nm']:
            data_settings.append((k, b, dtype[k]))
    print "Data Created"
    db = LoaderBoss(str(pid),inst_q,in_dir,data_settings)
    pb = PackerBoss(str(pid), results_q, odir, (dsize['rms'], dtype['rms']) )
    db.start()
    pb.start()
    db.set_add_data()
    ctr = 0
    t = []
    prev_time = time.time()
    while True:
        print time.time() - prev_time
        prev_time = time.time()
        print "*"*10 +str(ctr) +"*"*10
        ready = db.wait_data_ready( time_out=5 )
        if ready:
            db.clear_data_ready()
            my_f = db.get_file_id()

            expression_matrix = db.get_expression_matrix()
            gene_map = db.get_gene_map()
            sample_map = db.get_sample_map()
            network_map = db.get_network_map()
            exp = data.SharedExpression( expression_matrix )
            exp.orig_nsamples = t_nsamp
            gm = data.SharedGeneMap( gene_map )
            sm = data.SharedSampleMap( sample_map )
            nm = data.SharedNetworkMap( network_map )
            srt,rt,rms =  sharedprocesses.runSharedDirac( exp, gm, sm, nm, sample_block_size, npairs_block_size, nets_block_size )
            """
            uncomment to test srt and rt
            srt.fromGPU()
            np.save(os.path.join(in_dir, 'srt_hacky_'+my_f['file_id']), srt.buffer_data)
            rt.fromGPU()
            np.save(os.path.join(in_dir, 'rt_hacky_'+my_f['file_id']), rt.buffer_data)
            """
            db.release_loader_data()
            db.set_add_data()
            while not pb.ready():
                print "z"
                time.sleep(.5)
            rms.fromGPU( pb.get_mem() )
            pb.set_meta( my_f['file_id'], ( rms.buffer_nnets, rms.buffer_nsamples ), dtype['rms'] ) 
            pb.release()
        else:
            if db.empty():
                break
            else:
                raise Exception("Stuck")
    logging.info( "Tester: no data, all processed, killing sp")
    db.kill_all()
    pb.kill()
    db.clean_up()
    pb.clean_up()
    all_match = True
    while not results_q.empty():
        my_dict = results_q.get()
        proc = np.load(os.path.join(in_dir, my_dict['f_name']))
        single = np.load(os.path.join(in_dir, 'rms_'+ my_dict['file_id'] + '_single.npy'))
        (a,b) = single.shape
        print "Comparing", os.path.join(in_dir, my_dict['f_name']), " and ", os.path.join(in_dir, 'rms_'+ my_dict['file_id'] + '_single.npy')
        match = np.allclose(proc[:a,:b], single)
        print "Matching",my_dict['file_id'], match
        if not match:
            all_match = False
    if all_match:
        print "All tests SUCCESSFUL"
    else:
        print "You have been weighed, you have been measured, and you have been found wanting."

    logging.info( "Tester: exitted gracefully")
    ctx.pop()


def testMulti(pid=0):
    in_dir = '/scratch/sgeadmin/'
    odir = '/scratch/sgeadmin/out'
    np_list = []
    sample_block_size = 32
    npairs_block_size = 16
    nets_block_size = 8

    inst_q = Queue()
    results_q = Queue()
    check_cp = False #check whether in == out, if False, comparing speed
    check_list = []
    unique_fid = set()
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
    dsize = {'em':0, 'gm':0, 'sm':0, 'nm':0, 'rms':0}
    dtype = {'em':np.float32, 'gm':np.int32, 'sm':np.int32,'nm':np.int32,'rms':np.float32 }

    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()

    dsize, check_list = addFakeDataQueue(unique_fid, in_dir, inst_q, check_list,dsize, sample_block_size, npairs_block_size, nets_block_size)
    data_settings = []
    for k,b in dsize.iteritems():
        if k in ['em','sm','gm','nm']:
            data_settings.append((k, b, dtype[k]))
    print "Data Created"
    #db = LoaderBoss(str(pid),inst_q,in_dir,data_settings)
    db_q = LoaderQueue( inst_q, in_dir, data_settings)
    pb_q = PackerQueue( results_q, odir, (dsize['rms'], dtype['rms']) )
    db_q.add_loader_boss(10)
    pb_q.add_packer_boss(10)
    #pb = PackerBoss(str(pid), results_q, odir, (dsize['rms'], dtype['rms']) )
    #db.start()
    #pb.start()
    #db.set_add_data()
    ctr = 0
    t = []
    prev_time = time.time()
    while True:
        print time.time() - prev_time
        prev_time = time.time()
        print "*"*10 +str(ctr) +"*"*10
        #ready = db.wait_data_ready( time_out=5 )
        try:
            db = db_q.next_loader_boss()
            ready = True
            md_count = 0
        except MaxDepth:
            if db_q.no_data():
                print "max depth and no data"
                break
            else:
                raise
        if ready:
            db.clear_data_ready()
            my_f = db.get_file_id()

            expression_matrix = db.get_expression_matrix()
            gene_map = db.get_gene_map()
            sample_map = db.get_sample_map()
            network_map = db.get_network_map()
            exp = data.SharedExpression( expression_matrix )
            gm = data.SharedGeneMap( gene_map )
            sm = data.SharedSampleMap( sample_map )
            nm = data.SharedNetworkMap( network_map )
            srt,rt,rms =  sharedprocesses.runSharedDirac( exp, gm, sm, nm, sample_block_size, npairs_block_size, nets_block_size )

            """
            uncomment to test srt and rt
            srt.fromGPU()
            np.save(os.path.join(in_dir, 'srt_hacky_'+my_f['file_id']), srt.buffer_data)
            rt.fromGPU()
            np.save(os.path.join(in_dir, 'rt_hacky_'+my_f['file_id']), rt.buffer_data)
            """
            db.release_loader_data()
            db.set_add_data()
            pb = pb_q.next_packer_boss()
            rms.fromGPU( pb.get_mem() )
            pb.set_meta( my_f['file_id'], ( rms.buffer_nnets, rms.buffer_nsamples ), dtype['rms'] ) 
            pb.release()
    logging.info( "Tester: no data, all processed, killing sp")
    db_q.kill_all()
    #db.kill_all()
    pb_q.kill_all()
    #db.clean_up()
   
    all_match = True
    while not results_q.empty():
        try:
            my_dict = results_q.get()
            proc = np.load(os.path.join(odir, my_dict['f_name']))
            single = np.load(os.path.join(in_dir, 'rms_'+ my_dict['file_id'] + '_single.npy'))
            (a,b) = single.shape
            print "Comparing", os.path.join(odir, my_dict['f_name']), " and ", os.path.join(in_dir, 'rms_'+ my_dict['file_id'] + '_single.npy')
            match = np.allclose(proc[:a,:b], single)
            print "Matching",my_dict['file_id'], match
            if not match:
                all_match = False
        except ValueError as e:
            print e

            
            all_match = False
    if all_match:
        print "All tests SUCCESSFUL"
    else:
        print "You have been weighed, you have been measured, and you have been found wanting."

    logging.info( "Tester: exitted gracefully")
    ctx.pop()


def addFakeDataQueue(unique_fid, in_dir,inst_q, check_list,dsize, sample_block_size, npairs_block_size, nets_block_size):
    
    for i in range(100):
        fake = genFakeData( 200, 20000)
        p_hash = None
        for i in range(1):
            f_dict = {}
            f_id = str(random.randint(10000,100000))
            while f_id in unique_fid:
                f_id = str(random.randint(10000,100000))
            f_dict['file_id'] = f_id
            unique_fid.add(f_id)

            for k,v in fake.iteritems():
                if k == 'em':
                    exp = data.Expression(v)
                    exp.createBuffer(sample_block_size, buff_dtype=np.float32)
                    v = exp.buffer_data
                    t_nsamp = exp.orig_nsamples
                elif k == 'sm':
                    sm = data.SampleMap(v)
                    sm.createBuffer(sample_block_size, buff_dtype=np.int32)
                    v = sm.buffer_data
                elif k == 'gm':
                    gm = data.GeneMap(v) 
                    gm.createBuffer( npairs_block_size, buff_dtype=np.int32)
                    v = gm.buffer_data
                elif k == 'nm':
                    nm = data.NetworkMap(v)
                    nm.createBuffer( nets_block_size, buff_dtype=np.int32 )
                    v = nm.buffer_data
                f_hash = hashlib.sha1(v).hexdigest()
                if k == 'em':
                    if p_hash is None:
                        p_hash = f_hash
                        p_temp = v.copy()
                    else:
                        assert p_hash == f_hash, str(v) + " " + str(p_temp)
                        p_hash = None
                        p_temp = None
                f_name = '_'.join([ k, f_dict['file_id'], f_hash])
                with open(os.path.join( in_dir, f_name),'wb') as f:
                    np.save(f, v)
                if v.size > dsize[k]:
                    dsize[k] = v.size
                f_dict[k] = f_name
            srt,rt,rms = processes.runDirac(exp.orig_data, gm.orig_data, sm.orig_data,nm.orig_data, sample_block_size, npairs_block_size, nets_block_size, True)
            """
            uncomment to compare srt and rts
            srt.fromGPU()
            np.save(os.path.join(in_dir, 'srt_'+ f_dict['file_id'] + '_single'), srt.res_data)
            rt.fromGPU()
            np.save(os.path.join(in_dir, 'rt_'+ f_dict['file_id'] + '_single' ), rt.res_data)
            """
            rms.fromGPU(res_dtype=np.float32)
            np.save(os.path.join(in_dir, 'rms_'+ f_dict['file_id'] + '_single'), rms.res_data)

            rms = data.RankMatchingScores( nm.buffer_nnets, sm.buffer_nsamples )
            
            rms.createBuffer(  sample_block_size, nets_block_size, buff_dtype=np.float32)
            if rms.buffer_data.size > dsize['rms']:
               dsize['rms'] = rms.buffer_data.size 
            inst_q.put( f_dict )
            check_list.append(f_dict)
    return dsize, check_list


def genFakeData( n, gn):
    neighbors = random.randint(5, 20)
    nnets = random.randint(50,100)

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
    tcdirac.debug.initLogging("tcdirac_gpu_mptest.log", logging.INFO, st_out=True)
    testMulti(1)

