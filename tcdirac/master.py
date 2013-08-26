import static
import debug
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
"""
def initLogging():
    comm = MPI.COMM_WORLD
    logfile = "/scratch/sgeadmin/log_mpi_r%i.txt"%comm.Get_rank()
    log_format = '%(asctime)s - %(name)s rank['+str( comm.Get_rank() )+']- %(levelname)s - %(message)s'
    logging.basicConfig(filename=loggfile, level=logging.INFO, format=log_format)

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

def makeDirs(dirs):
    comm = MPI.COMM_WORLD
    if isHostBoss(comm):
        logging.info('Boss of %s'%socket.gethostname())
        for d in dirs:
            host_boss = True
            if not op.exists(d):
                os.makedirs(d)
    comm.barrier()

def checkDebug():
    if debug.DEBUG:
        logging.info('***DEBUG ON***')
        makeDirs([debug.debug_dir])        
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
    ""
    Host Boss is the smallest rank on a given host
    This is meant to figure out who does io.
   ""
    myh = socket.gethostname()
    myr = comm.Get_rank()
   
    hlist=comm.gather((myh,myr))
    hlist = comm.bcast(hlist) 
    
    for host, rank in hlist:
        if host == myh and rank < myr:
            return False
    return True
"""

def kNearest(compare_list,samp_name, samp_age, k):
    """
    Given compare_list, which contains tuples in sorted order
        of (sample_age, sample_name).
    returns k sample names that are closest in age to samp_age
    """
    off = k/2
    i = bisect.bisect_left(compare_list,(age,samp) )
    l = i - off
    u = i + off
    if l < 0:
        u = u - l
        l = 0
    if u >= len(compare_list):
        l = l - (u - (len(compare_list) - 1))
        u = len(compare_list) - 1

    samp_compare = [s for a,s in compare_list[l:u+1]]
    return samp_compare

def partitionSamplesByAllele( alleles, mi, cstrain):
    """
    Get a dictionary of lists of sample names and ages partitioned by allele in increasing
        age order.
    Given alleles(list of strings), mi(metadataInfo object), cstrain (string: current strain)
    returns dict[allele] -> list:[(age_1,samp_name_1), ... ,(age_n,samp_name_n)] sorted by age
    """
    samples = {}
    for allele in alleles:
        samples[allele] = [(mi.getAge(sample),sample)for sample in mi.getSampleIDs(cstrain,allele)]
        samples[allele].sort()
    return samples

def getSRTSByAllele(alleles,pw,samples):
    """
    Returns a dict containing sample rank templates for each alleles given the pathway(pw)
    """
    srts = {}
    for allele in alleles:
        srts[allele] = dirac.getSRT(sd.getExpression(pw,[s for a,s in samples[allele]]))
    return srts

def genRMS(comm,sd,mi,k_neighbors):
   
    #get and distribute pws and strains 
    pws = None
    if comm.rank == 0:
        pws = sd.getPathways()
        strain_list = mi.getStrains()

    pws = comm.bcast(pws)
    strain_list = comm.bcast(strain_list)

    
    for cstrain in strain_list:

        logging.info('Starting strain [%s]' % cstrain)
        
        mypws = [pw for i,pw in enumerate(pws) if i%comm.size == comm.rank]
        alleles = mi.getNominalAlleles(cstrain)
        indexes = ["%s_%s" % (pw,allele) for pw,allele in  itertools.product(mypws,alleles)]
        samples = mi.getSampleIDs(cstrain)

        #preallocate results dataframe 
        results = pandas.DataFrame(np.empty((len(indexes), len(samples)), dtype=float), index=indexes, columns=samples)
        for pw in mypws:
            #partition samples by strain/allele
            samples = partitionSamplesByAllele( alleles, mi, cstrain)

            #generate pw srts for all samples partitioned by strain/allele
            srts = getSRTSByAllele(alleles,pw,samples)

            for allele_base in alleles:
                for allele_compare in alleles:
                    r_index = "%s_%s" % (pw,allele_compare)
                    #list of samples with comparison allele
                    compare_list = samples[allele_compare]
                    for age, samp in samples[allele_base]:

                        samp_compare = kNearest(compare_list,samp_name, samp_age, k_neighbors)

                        comp_exp = srts[allele_compare].loc[:,samp_compare]
                        rt = dirac.getRT(comp_exp)
                        results[samp][r_index] =  dirac.getRMS(srts[allele_base][samp],rt)

        comm.barrier()
        return results

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
   
    initLogging()

    logging.info('Process starting')

    working_dir = '/scratch/sgeadmin/hddata/'
    working_bucket = 'hd_working_0'
    data_source_bucket = 'hd_source_data'

    k_neighbors = 5

    host_boss = False

    makeDirs([working_dir])
    getFiles()
    sd, mi = initData(comm)

    logging.info('Process ending')
