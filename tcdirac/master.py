from mpi4py import MPI
import data
import dirac
import logging
import socket
import os.path as op
import os
import boto
import bisect
import itertools
import pandas
import numpy as np

def initLogging():
    comm = MPI.COMM_WORLD
    configfile = "/scratch/sgeadmin/log_mpi_r%i.txt"%comm.Get_rank()
    log_format = '%(asctime)s - %(name)s rank['+str( comm.Get_rank() )+']- %(levelname)s - %(message)s'
    logging.basicConfig(filename=configfile, level=logging.INFO, format=log_format)

def getFiles():
    if comm.Get_rank() == 0:
        if not op.exists(op.join( working_dir,'metadata.txt')):
            conn = boto.connect_s3()
            b = conn.get_bucket(data_source_bucket)
            k.key = 'metadata.txt'
            k.get_content_to_filename(op.join( working_dir,'metadata.txt'))
        
    if comm.Get_rank() == 1:
        if not op.exists(op.join( working_dir, 'trimmed_dataframe.pandas')):
            conn = boto.connect_s3()
            b = conn.get_bucket(data_source_bucket)
            k.key ='trimmed_dataframe.pandas'
            k.get_content_to_filename(op.join( working_dir,'trimmed_dataframe.pandas'))

def makeDirs():
    comm = MPI.COMM_WORLD
    if isHostBoss(comm):
        #print socket.gethostname(),comm.Get_rank(), "loves it when you call it big papa"
        logging.info('Boss of %s'%socket.gethostname())
        host_boss = True
        if not op.exists(working_dir):
            os.makedirs(working_dir)

def initData(comm):

    sd = data.SourceData()
    mi = None
    if comm.Get_rank() == 0:
        logging.info('init SourceData')
        sd.load_dataframe()
        sd.load_net_info()
        logging.info('init MetaInfo')
        mi = data.MetaInfo(op.join(working_dir,'metadata.txt'))
    logging.info("Broadcasting SourceData and MetaInfo")
    sd = comm.bcast(sd)
    mi = comm.bcast(mi)
    assert(sd.source_dataframe is not None)
    logging.info("Received SourceData and MetaInfo")

    return sd, mi

def isHostBoss(comm):
    """
    Host Boss is the smallest rank on a given host
    This is meant to figure out who does io.
    """
    myh = socket.gethostname()
    myr = comm.Get_rank()
   
    hlist=comm.gather((myh,myr))
    hlist = comm.bcast(hlist) 
    
    for host, rank in hlist:
        if host == myh and rank < myr:
            return False
    return True

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
   
    initLogging()

    logging.info('Process starting')

    working_dir = '/scratch/sgeadmin/hddata/'
    working_bucket = 'hd_working_0'
    data_source_bucket = 'hd_source_data'

    k_neighbors = 5

    host_boss = False

    makeDirs()
    getFiles()
    sd, mi = initData(comm)
    cstrain = None
    strain_id = -1 
    while True:
        if comm.rank == 0:
            strain_list = mi.getStrains()
            strain_id += 1
            if strain_id < len(strain_list):
                cstrain = strain_list[strain_id]
            else:
                cstrain = 'STOP'    

        cstrain = comm.bcast(cstrain)
        if cstrain == 'STOP':
            logging.info("Received STOP command")
            break

        logging.info('Starting strain [%s]' % cstrain)
        pws = sd.getPathways()
        mypws = [pw for i,pw in enumerate(pws) if i%comm.size == comm.rank]
        alleles = mi.getNominalAlleles(cstrain)
        indexes = ["%s_%s" % (pw,allele) for pw,allele in  itertools.product(mypws,alleles)]
        samples = mi.getSampleIDs(cstrain)
        
        results = pandas.DataFrame(np.empty((len(indexes), len(samples)), dtype=float), index=indexes, columns=samples)
        for pw in mypws:
            samples = {}
            for allele in alleles:
                samples[allele] = [(mi.getAge(sample),sample)for sample in mi.getSampleIDs(cstrain,allele)]
                samples[allele].sort()


            off = k_neighbors/2
            srts = {}
            for allele in alleles:
                srts[allele] = dirac.getSRT(sd.getExpression(pw,[s for a,s in samples[allele]]))
            
            for allele_base in alleles:
                for allele_compare in alleles:
                    r_index = "%s_%s" % (pw,allele_compare)
                    for age, samp in samples[allele_base]:
                        i = bisect.bisect(samples[allele_compare],(age,samp) )
                        l = i - off
                        u = i + off
                        if l < 0:
                            u = u - l
                            l = 0
                        if u >= len(samples[allele_compare]):
                            l = l - (u - (len(samples[allele_compare]) - 1))
                            u = len(samples[allele_compare]) - 1
                        samp_compare = [s for a,s in samples[allele_compare][l:u+1]]
                        comp_exp = srts[allele_compare].loc[:,samp_compare]
                        rt = dirac.getRT(comp_exp)
                        results[samp][r_index] =  dirac.getRMS(srts[allele_base][samp],rt)

        comm.barrier()
        results.to_pickle(op.join(working_dir, 'rms.%s.%i.pandas'%(cstrain, comm.Get_rank()))) 

    logging.info('Process ending')
