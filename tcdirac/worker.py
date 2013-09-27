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
import time
from profiler import MPIProfiler
import random
from boto.s3.key import Key
import boto
import copy
from boto.exception import S3ResponseError
class Worker:
    """
    tcdirac worker class
    """
    def __init__(self, comm, working_dir, working_bucket, ds_bucket, logging_level=logging.INFO):
        self._comm = comm
        #create a communicators specific to each host
        self._host_comm, self._host_map = self._hostConfig()

        self._initLogging(logging_level)

        self._datasource_bucket = ds_bucket
        self._working_dir = working_dir
        self._working_bucket = working_bucket
        logging.info("Initial settings: working_dir[%s], working bucket [%s], datasource bucket [%s]" 
            % (working_dir, working_bucket, ds_bucket))

        #variables set elseware
        self._nominal_alleles = {}
        self._sd = None #data.SourceData object
        self._mi = None #data.MetaInfo
        self._results = {}
        self._srt_cache = {}
        self._sample_x_allele = {}
        self._pathways = None
        self._initData()

    def _initLogging(self, level=logging.INFO):
        """
        Initialize logging
        """
        comm = MPI.COMM_WORLD
        logfile = "/scratch/sgeadmin/log_mpi_r%i.txt"%comm.Get_rank()
        log_format = '%(asctime)s - %(name)s rank['+str( comm.Get_rank() )+']- %(levelname)s - %(message)s'
        logging.basicConfig(filename=logfile, level=level, format=log_format)

    def _initData(self, data_master=0):
        """
        Parses data (sourcedata and metainfo) and distributes it to all nodes in comm
        """
        comm = self._comm

        self.makeDirs([self._working_dir])
        self._getDataFiles(data_master)
        sd = data.SourceData()
        mi = None
        if comm.rank == data_master:
            logging.info('init SourceData')
            sd.load_dataframe()
            sd.load_net_info()
            logging.info('init MetaInfo')
            mi = data.MetaInfo(op.join(self._working_dir,'metadata.txt'))
        logging.info("Broadcasting SourceData and MetaInfo")
        sd = comm.bcast(sd)
        mi = comm.bcast(mi)
        logging.info("Received SourceData and MetaInfo")
     
        self._sd = sd
        self._mi = mi

    def _getDataFiles(self,file_master=0):
        """
        Retrieves metadata and parsed dataframe files
            (generated by utilities/hddata_process.py) from S3
        """
        comm = self._comm
        working_dir = self._working_dir
        data_source_bucket = self._datasource_bucket
       
        if comm.rank == file_master:
            if not op.exists(op.join( working_dir,'metadata.txt')):
                conn = boto.connect_s3()
                b = conn.get_bucket(data_source_bucket)
                k = Key(b)
                k.key = 'metadata.txt'
                k.get_contents_to_filename(op.join( working_dir,'metadata.txt'))

        if comm.rank == file_master:
            if not op.exists(op.join( working_dir, 'trimmed_dataframe.pandas')):
                conn = boto.connect_s3()
                b = conn.get_bucket(self._working_bucket)
                k = Key(b)
                k.key ='trimmed_dataframe.pandas'
                k.get_contents_to_filename(op.join( working_dir,'trimmed_dataframe.pandas'))
        comm.barrier()

    def _hostConfig(self):
        """
        Creates a host communicator object and a map from hostnames to hostmasters
        """
        comm = self._comm

        myh = socket.gethostname()
        myr = comm.rank
       
        hlist=comm.gather((myh,myr))
        
        hlist = comm.bcast(hlist) 
        hlist.sort()
        hm = self._host_map = {}
        h_counter = 0
        
        for h,r in hlist:
            if h not in hm:
                hm[h] = (r,h_counter)
                #smallest rank in host is boss
                h_counter += 1
        host_comm = comm.Split( hm[myh][1] )
        host_comm.name = myh

        if host_comm.rank == 0:
            #check that our host master is equal to
            #what we expect
            assert(hm[myh][0] == self._comm.rank)
        return (host_comm, hm)
        

    def isHostMaster(self):
        """
        Returns true if is the smallest rank on the host machine
        """
        return self._host_comm.rank == 0

    def checkDebug(self):
        """
        Prepares host for debug mode if required
        """
        if debug.DEBUG:
            logging.info('***DEBUG ON***')
            makeDirs([debug.debug_dir])


    def makeDirs(self, dirs):
        comm = self._comm
        if self.isHostMaster():
            logging.info('Boss of %s'%socket.gethostname())
            for d in dirs:
                if not op.exists(d):
                    logging.info("Creating [%s]"%d)
                    os.makedirs(d)
        comm.barrier()

       
    def kNearest(self,compare_list,samp_name, samp_age, k):
        """
        Given compare_list, which contains tuples in sorted order
            of (sample_age, sample_name).
        returns k sample names that are closest in age to samp_age
        """
        compare_list = [(a,n) for a,n in compare_list if (a,n) != (samp_age,samp_name)]
        while k > len(compare_list):
            logging.warning("k is too large [%i], adjusting to [%i]"%(k,k-1))
            k -= 1

        off = k/2
        i = bisect.bisect_left(compare_list,(samp_age,samp_name) )
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

    def getAlleles(self, cstrain):
        if cstrain not in self._nominal_alleles:
            self._nominal_alleles[cstrain] = self._mi.getNominalAlleles(cstrain)
        return self._nominal_alleles[cstrain]

    def getStrains(self, comm=None, strain_master=0):
        return self._mi.getStrains()
 
    def _getPathways(self, comm=None, pathway_master=0):
        pws = None
        if comm is None:
            comm = self._comm
        if comm.rank == pathway_master:
            pws = self._sd.getPathways()
            #dbase hit, so limit conns
            #pws.sort()
        return comm.bcast(pws, root=pathway_master)

    def _getSamplesByStrain(self, strain):
        """
        Returns a list of all sample ids belonging to strain
        """
        return self._mi.getSampleIDs(strain)

    def getMyPathways(self,comm=None):
        if comm == None:
            comm = self._comm 
        if self._pathways is None:
            pathways = self._pathways = self._getPathways( comm )
        else:
            pathways = self._pathways
        return [pw for i,pw in enumerate(pathways) if i%comm.size == comm.rank]

    def initRMSDFrame(self,my_pathways, cstrain, comm=None):
        if comm==None:
           comm = self._comm
        alleles = self.getAlleles(cstrain)
        samples = self._getSamplesByStrain(cstrain)
        indexes = ["%s_%s" % (pw,allele) for pw,allele in  itertools.product(my_pathways, alleles)]
        return pandas.DataFrame(np.empty(( len(indexes), len(samples)), dtype=float), index=indexes, columns=samples)


    def _partitionSamplesByAllele(self, cstrain, shuffle=False):
        """
        Get a dictionary of lists of sample names and ages partitioned by allele in increasing
            age order.
        Given alleles(list of strings), mi(metadataInfo object), cstrain (string: current strain)
        returns dict[allele] -> list:[(age_1,samp_name_1), ... ,(age_n,samp_name_n)] sorted by age
        """
        mi = self._mi

        samples = {}
        if not shuffle:
            for allele in self.getAlleles(cstrain):
                samples[allele] = [(mi.getAge(sample),sample)for sample in mi.getSampleIDs(cstrain,allele)]
                samples[allele].sort()
        else:
            #test
            old = copy.deepcopy(self._sample_x_allele)
            all_samples = mi.getSampleIDs(cstrain)[:]
            random.shuffle(all_samples)
            for allele in self.getAlleles(cstrain):
                n = len(mi.getSampleIDs(cstrain,allele))
                samples[allele] = [(mi.getAge(s),s) for s in all_samples[:n]]
                samples[allele].sort()
                all_samples = all_samples[n:]
        self._sample_x_allele[cstrain] = samples 


    def initStrain(self, cstrain, mypws, shuffle=False):
        self._cstrain = cstrain
        self._partitionSamplesByAllele(cstrain, shuffle)
        self._results[cstrain] = self.initRMSDFrame( mypws, cstrain )

    def getSamplesByAllele(self, cstrain, allele):
        return self._sample_x_allele[cstrain][allele]

    def genSRTs(self, cstrain, pw):
        #self.p.start("genSRTs")
        if (cstrain, pw) not in self._srt_cache:
          
            srts = None 
            sd = self._sd
            mi = self._mi
            samples = []
            for allele in self.getAlleles(cstrain):
                samples += [s for a,s in self._sample_x_allele[cstrain][allele]]
                #self.p.start("getExpression")
            expFrame = sd.getExpression( pw, samples)
            srts = dirac.getSRT( expFrame )
            self._srt_cache[(cstrain,pw)] = srts

        return self._srt_cache[(cstrain, pw)]

    def getRMS(self, rt, srt):
        return dirac.getRMS( srt, rt )

    def setRMS(self, rms, index, samp):
        self._results[self._cstrain][samp][index] = rms

    def saveRMS(self,prefix='rms'):
        for strain, table in self._results.iteritems():
            ofile_name = '%s.%s.%i.pandas.pkl' % (prefix,strain,self._comm.rank)
            table.to_pickle(op.join(self._working_dir,ofile_name))

    def classify( self, comm=None ):
        
        if comm is None:
            comm = self._comm
        class_dict = {}
        for strain in self.getStrains():
            mypws = self.getMyPathways(comm)
            res = self._results[strain]
            class_dict[strain] = pandas.DataFrame(np.empty(( len(mypws), len(res.columns)), dtype=int) , index=mypws, columns = res.columns )
            for pw in mypws:
                alleles = self.getAlleles(strain)
                for b_allele in alleles:
                    samps = [s for a,s in self.getSamplesByAllele(strain, b_allele)]

                    b_rows = ["%s_%s" %(pw,allele) for allele in alleles if allele != b_allele]
                    
                    for samp in samps:
                        class_dict[strain][samp][pw] = 1
                        for row in b_rows:
                            if res.loc[row,samp] >= res.loc["%s_%s" % (pw, b_allele), samp]:
                                class_dict[strain][samp][pw] = 0
            class_dict[strain] = class_dict[strain].sum(1)
        self._classification_res = class_dict
        return class_dict

    def runBase(self, k_neighbors):
        wkr = self
        logging.info("running base")
        for cstrain in wkr.getStrains():
            logging.info("Strain[%s] starting" % cstrain)
            mypws = wkr.getMyPathways()
            alleles = wkr.getAlleles(cstrain)
            wkr.initStrain(cstrain, mypws, shuffle=False)
            for pw in mypws:
                rt_cache = {} 
                srts = wkr.genSRTs( cstrain, pw )
                for a_base, a_compare in itertools.product(alleles,alleles):
                    r_index = "%s_%s" % (pw, a_compare)
                    base_samp = wkr.getSamplesByAllele(cstrain, a_base)
                    comp_samp = wkr.getSamplesByAllele(cstrain,a_compare)
                    for age, samp in base_samp:
                        neighbors = wkr.kNearest(comp_samp, samp, age, k_neighbors)
                        nhash = ''.join(neighbors)
                        if nhash not in rt_cache:
                            srt_comp = srts.loc[:,neighbors]
                            rt = dirac.getRT(srt_comp)
                            rt_cache[nhash] = rt
                        else:
                            rt = rt_cache[nhash]
                        samp_srt = srts[samp]
                        rms = wkr.getRMS( rt, samp_srt ) 
                        wkr.setRMS(rms, r_index, samp)
        c = wkr.classify()
        return c

    def runPerm(self, num_runs, k_neighbors,truth):
        wkr = self
        c_results = {}
        for k,v in truth.iteritems():
            c_results[k] = v.copy()
            for i in v.index:
                c_results[k][i] = 0  
            
        test = []
        for ctr in range(num_runs): 
            temp = {}
            check = True
            for cstrain in wkr.getStrains():
                logging.info("Strain[%s] starting" % cstrain)
                mypws = wkr.getMyPathways()
                alleles = wkr.getAlleles(cstrain)
                wkr.initStrain(cstrain, mypws, shuffle=True)
                for pw in mypws:
                    rt_cache = {} 
                    srts = wkr.genSRTs( cstrain, pw )
                    for a_base, a_compare in itertools.product(alleles,alleles):
                        r_index = "%s_%s" % (pw, a_compare)
                        base_samp = wkr.getSamplesByAllele(cstrain, a_base)
                        #testing shuffle
                        temp[(pw,cstrain,a_base)] = base_samp
                        if len(test) > 0:
                            same = True
                            for x in temp[(pw,cstrain,a_base)]:
                                if x not in test[-1][(pw,cstrain,a_base)]:
                                    same = False
                            msg = 'In runperm\n'+ ''.join(map(str, temp[(pw,cstrain,a_base)]))+ '\n and \n '+''.join( map(str,test[-1][(pw,cstrain,a_base)]) )
                            assert same == False, msg

                        if check and self._comm.rank == 0:
                            print ctr,a_base, base_samp[:5]
                        comp_samp = wkr.getSamplesByAllele(cstrain,a_compare)
                        for age, samp in base_samp:
                            neighbors = wkr.kNearest(comp_samp, samp, age, k_neighbors)
                            nhash = ''.join(neighbors)
                            if nhash not in rt_cache:
                                srt_comp = srts.loc[:,neighbors]
                                rt = dirac.getRT(srt_comp)
                                rt_cache[nhash] = rt
                            else:
                                rt = rt_cache[nhash]
                            samp_srt = srts[samp]
                            rms = wkr.getRMS( rt, samp_srt ) 
                            wkr.setRMS(rms, r_index, samp)
                    check = False
            test.append(temp)
            c = wkr.classify()
            
            for key in c.keys():
                for i in c[key].index:
                    if truth[key][i] <= c[key][i]:
                        msg = "key: [%s] index[%s] ctr[%i] value[%i]" %(key,i,ctr,c_results[key][i])
                        assert c_results[key][i] <= ctr, msg
                        c_results[key][i] += 1
                        
                truth[key].to_pickle('/scratch/sgeadmin/unjoined.truth.perm.%s.df.%i.%i.pkl'%(key,self._comm.rank,ctr))
                c[key].to_pickle('/scratch/sgeadmin/unjoined.perm.%s.df.%i.%i.pkl'%(key,self._comm.rank,ctr))
        return c_results       

    def joinResults(self, results):
        all_results = None
        if self._comm.rank == 0:
            pws = self._sd.getPathways()
            strains = self.getStrains()
            comb_results =  pandas.DataFrame(np.zeros((len(pws), len(strains)), dtype=int), index=pws,columns=strains)
        all_results = self._comm.gather(results)
        
        if self._comm.rank == 0:
            for r in all_results:
                for strain,series in r.iteritems(): 
                    for i in series.index:
                        comb_results[strain][i] = series[i]
            return comb_results
        return None
        

class NodeFactory:
    def __init__(self, world_comm):
        host_comm, type_comm, master_comm = self.genComms(world_comm)
        
        if master_comm == MPI.COMM_NULL:
            #worker node
            if type_comm.name == 'gpu':
                self.thisNode = GPUNode(world_comm, host_comm, type_comm, master_comm)
            else:
                self.thisNode = DataNode(world_comm, host_comm, type_comm, master_comm)
        else:

            if type_comm.name == 'gpu':
                self.thisNode = MasterGPUNode(world_comm, host_comm, type_comm, master_comm)
            else:
                self.thisNode = MasterDataNode(world_comm, host_comm, type_comm, master_comm)

    def getNode(self):
        return self.thisNode

    def genComms(self, world_comm):
        host_comm = self.genHostComm(world_comm)
        type_comm = self.genTypeComm(world_comm)
        master_comm = self.genMasterComm(world_comm, type_comm, host_comm)

        return (host_comm, type_comm, master_comm)



    def genTypeComm(self, world_comm):
        isgpu = True
        if socket.gethostname() == 'master':
            isgpu = False
        type_comm = world_comm.Split(0 if isgpu else 1)
        if isgpu:
            type_comm.name = 'gpu'
        else:
            type_comm.name = 'data'
        return type_comm


    def genHostComm(self, world_comm):
        
        myh = socket.gethostname()
        myr = world_comm.rank
       
        hlist=world_comm.gather((myh,myr))
        
        hlist = world_comm.bcast(hlist) 
        hlist.sort()
        hm = {}
        h_counter = 0
        
        for h,r in hlist:
            if h not in hm:
                hm[h] = (r,h_counter)
                #smallest rank in host is boss
                h_counter += 1
        host_comm = world_comm.Split( hm[myh][1] )
        host_comm.name = myh
        return host_comm

    def genMasterComm(self, world_comm, type_comm, host_comm):
        master = False
        if type_comm.name == 'gpu' and host_comm.rank == 0:
            master = True
        if type_comm.name == 'data' and type_comm.rank == 0:
            master = True
        results = np.empty((world_comm.size,), dtype=int)
        world_comm.Allgather(np.array([0 if not master else 1]), results)
        not_masters = [i for i in range(world_comm.size) if results[i] == 0]

        world_group = world_comm.Get_group()
        master_group = world_group.Excl(not_masters)

        master_comm = world_comm.Create(master_group)
        if master_comm:
            master_comm.name = 'master'

        return master_comm

class HeteroNodeFactory(NodeFactory):

    def __init__(self, world_comm, isgpu):
        self.isgpu = isgpu
        NodeFactory.__init__(self, world_comm)

    def genTypeComm(self, world_comm):
        sgpu = self.isgpu

        type_comm = world_comm.Split(0 if isgpu else 1)
        if isgpu:
            type_comm.name = 'gpu'
        else:
            type_comm.name = 'data'
        return type_comm

class MPINode:
    def __init__(self, world_comm, host_comm, type_comm, master_comm ):
        self.world_comm = world_comm
        self.host_comm = host_comm
        self.type_comm = type_comm
        self.master_comm = master_comm
        self.log_init()


    def log_init(self):
        logging.info( "world comm rank:\t %i" % self.world_comm.rank )
        logging.info( "host name:\t %s" % self.host_comm.name )
        logging.info( "host rank:\t %i" % self.host_comm.rank )
        logging.info( "type name:\t %s" % self.type_comm.name)
        logging.info( "type rank:\t %i " % self.type_comm.rank)
        if self.master_comm == MPI.COMM_NULL:
            logging.info( "Not master")
        else:
            logging.info( "master rank:\t %i" % self.master_comm.rank)

    def nodeType(self):
        return self.type_comm.name

    def hostMaster(self):
        return self.host_comm.rank == 0

    def masterNode(self):
        return  self.master_comm != MPI.COMM_NULL

    def makeDirs(self, dirs, force=False):
        if self.hostMaster() or force :
            for d in dirs:
                if not op.exists(d):
                    logging.info("Creating [%s]"%d)
                    os.makedirs(d)

    def getData(self, working_dir, working_bucket, ds_bucket):
        pass

class DataNode(MPINode):
    def __init__(self, world_comm, host_comm, type_comm, master_comm ):
        MPINode.__init__(self, world_comm, host_comm, type_comm, master_comm )   

    def getData(self, working_dir, working_bucket, ds_bucket):
        logging.debug("Getting sourcedata and metainfo")
        self._sd = self.type_comm.bcast(None)
        self._mi = self.type_comm.bcast(None)
        logging.debug("Received SourceData and MetaInfo")

    

class MasterDataNode(DataNode):
    def __init__(self, world_comm, host_comm, type_comm, master_comm ):
        DataNode.__init__(self, world_comm, host_comm, type_comm, master_comm )   

    def getData(self, working_dir, working_bucket, ds_bucket):
        self.working_dir = working_dir
        self.working_bucket = working_bucket
        self.ds_bucket = ds_bucket

        self.makeDirs([self.working_dir])
        self._getDataFiles()
        sd = data.SourceData()
        mi = None
        logging.info('init SourceData')
        sd.load_dataframe()
        sd.load_net_info()
        logging.info('init MetaInfo')
        mi = data.MetaInfo(op.join(self.working_dir,'metadata.txt'))
        logging.info("Broadcasting SourceData and MetaInfo")
        sd = self.type_comm.bcast(sd)
        mi = self.type_comm.bcast(mi)
        logging.info("Received SourceData and MetaInfo")

        self._sd = sd
        self._mi = mi


    def _getDataFiles(self):
        """
        Retrieves metadata and parsed dataframe files
            (generated by utilities/hddata_process.py) from S3
        """
        working_dir = self.working_dir
        data_source_bucket = self.ds_bucket
   
        if not op.exists(op.join( working_dir,'metadata.txt')):
            conn = boto.connect_s3()
            b = conn.get_bucket(data_source_bucket)
            k = Key(b)
            k.key = 'metadata.txt'
            k.get_contents_to_filename(op.join( working_dir,'metadata.txt'))

        if not op.exists(op.join( working_dir, 'trimmed_dataframe.pandas')):
            conn = boto.connect_s3()
            try:
                b = conn.get_bucket(self.working_bucket)
                k = Key(b)
                k.key ='trimmed_dataframe.pandas'
                k.get_contents_to_filename(op.join( working_dir,'trimmed_dataframe.pandas'))
            except S3ResponseError:
                print "Have you run ~/hdproject/utilities/hddata_process.py lately"
                raise

class GPUNode(MPINode):
    def __init__(self, world_comm, host_comm, type_comm, master_comm ):
        MPINode.__init__(self, world_comm, host_comm, type_comm, master_comm )   

class MasterGPUNode(GPUNode):
    def __init__(self, world_comm, host_comm, type_comm, master_comm ):
        GPUNode.__init__(self, world_comm, host_comm, type_comm, master_comm )   


   

if __name__ == "__main__":
    import time
    worker_settings = {
                'working_dir':'/scratch/sgeadmin/hddata/', 
                'working_bucket':'hd_working_0', 
                'ds_bucket':'hd_source_data', 
                }
    world_comm = MPI.COMM_WORLD
    level = logging.INFO
    isgpu = True
    try:
        import pycuda.driver as cuda
        from gpu import processes
    except ImportError:
        isgpu = False
    
    logfile = "/scratch/sgeadmin/log_mpi_r%i.txt"%world_comm.Get_rank()
    log_format = '%(asctime)s - %(name)s rank['+str( world_comm.Get_rank() )+']- %(levelname)s - %(message)s'
    logging.basicConfig(filename=logfile, level=level, format=log_format)
    logging.info("Starting")

    nf = HeteroNodeFactory( world_comm, isgpu )
    thisNode = nf.getNode()
    thisNode.getData(**worker_settings)

    logging.info("Exiting")
    
    
    """
    me = None
    if host_name == 'master':
        if world_comm.rank == 0:
            me = MasterNode(host_name,
        me = DataNode()
        
    else:
        if host_comm.rank == 0:
            me = GPUMaster( host_name,host_comm, host_map )
        else:
            me = GPUWorker( host_name, host_comm, host_map )

    worker_settings = {
                'comm':MPI.COMM_WORLD, 
                'working_dir':'/scratch/sgeadmin/hddata/', 
                'working_bucket':'hd_working_0', 
                'ds_bucket':'hd_source_data', 
                'logging_level':logging.INFO
                }
    k_neighbors = 20
    num_runs=50
    class_acc = []
    #a dictionary to hold the result dataframes, keyed by strain
    results = {}

    wkr = Worker(**worker_settings) 
    truth = wkr.runBase(k_neighbors)
    r = wkr.runPerm(num_runs, k_neighbors, truth)
    for key in r.keys():
        r[key].to_pickle('/scratch/sgeadmin/results.%s.df.%i.pkl'%(key,world_comm.rank) )
    comb_results = wkr.joinResults(r)
    if world_comm.rank == 0:
        comb_results.to_pickle('/scratch/sgeadmin/comb_results.%s.pkl' % k_neighbors)
    for ctr in range(num_runs): 
        for cstrain in wkr.getStrains():
            logging.info("Strain[%s] starting" % cstrain)
            mypws = wkr.getMyPathways()
            alleles = wkr.getAlleles(cstrain)
            wkr.initStrain(cstrain, mypws, shuffle=True)
            for pw in mypws:
                rt_cache = {} 
                srts = wkr.genSRTs( cstrain, pw )
                for a_base, a_compare in itertools.product(alleles,alleles):
                    r_index = "%s_%s" % (pw, a_compare)
                    base_samp = wkr.getSamplesByAllele(cstrain, a_base)
                    comp_samp = wkr.getSamplesByAllele(cstrain,a_compare)
                    for age, samp in base_samp:
                        neighbors = wkr.kNearest(comp_samp, samp, age, k_neighbors)
                        nhash = ''.join(neighbors)
                        if nhash not in rt_cache:
                            srt_comp = srts[a_compare].loc[:,neighbors]
                            rt = dirac.getRT(srt_comp)
                            rt_cache[nhash] = rt
                        else:
                            rt = rt_cache[nhash]
                        samp_srt = srts[a_base][samp]
                        rms = wkr.getRMS( rt, samp_srt ) 
                        wkr.setRMS(rms, r_index, samp)
        c = wkr.classify()
        class_acc.append(c)"""

