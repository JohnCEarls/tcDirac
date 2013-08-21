from mpi4py import MPI
import logging
import data
import debug
import os.path as op
class Worker:
    """
    tcdirac worker class
    """
    def __init__(self, comm, working_dir, working_bucket, ds_bucket, logging_level=logging.INFO):
        self._comm = comm
        self._initLogging(logging_level)

        self._datasource_bucket = ds_bucket
        self._working_dir = working_dir
        self._working_bucket = working_bucket
        self._sd = None #data.SourceData object
        self._mi = None #data.MetaInfo
        self._host_master = self._HostMaster()
        self.makeDirs([working_dir])
        self.getFiles()
        self.initData()

    def initData(self, data_master=0):
        sd = data.SourceData()
        mi = None
        if comm.Get_rank() == data_master:
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
     
        self._sd = sd
        self._mi = mi

    def _hostMaster(self):
        """
        Host Boss is the smallest rank on a given host
        This is meant to figure out who does io.
        """
        comm = self._comm

        myh = socket.gethostname()
        myr = comm.Get_rank()
       
        hlist=comm.gather((myh,myr))
        hlist = comm.bcast(hlist) 
        
        for host, rank in hlist:
            if host == myh and rank < myr:
                myr = rank
        return myr 

    def isHostMaster(self):
        return self._comm.rank == self._host_master

    def checkDebug(self):
        if debug.DEBUG:
            logging.info('***DEBUG ON***')
            makeDirs([debug.debug_dir])

        
    def _initLogging(self, level=logging.INFO):
        comm = MPI.COMM_WORLD
        logfile = "/scratch/sgeadmin/log_mpi_r%i.txt"%comm.Get_rank()
        log_format = '%(asctime)s - %(name)s rank['+str( comm.Get_rank() )+']- %(levelname)s - %(message)s'
        logging.basicConfig(filename=logfile, level=level, format=log_format)


    def getFiles(self,file_master=0):
        comm = self._comm
        working_dir = self._working_dir
        data_source_bucket = self._data_source_bucket

        if comm.rank == file_master:
            if not op.exists(op.join( working_dir,'metadata.txt')):
                conn = boto.connect_s3()
                b = conn.get_bucket(data_source_bucket)
                k.key = 'metadata.txt'
                k.get_content_to_filename(op.join( working_dir,'metadata.txt'))

        if comm.rank == file_master:
            if not op.exists(op.join( working_dir, 'trimmed_dataframe.pandas')):
                conn = boto.connect_s3()
                b = conn.get_bucket(data_source_bucket)
                k.key ='trimmed_dataframe.pandas'
                k.get_content_to_filename(op.join( working_dir,'trimmed_dataframe.pandas'))
        comm.barrier()

    def makeDirs(self, dirs):
        comm = self._comm
        if self.isHostMaster():
            logging.info('Boss of %s'%socket.gethostname())
            for d in dirs:
                if not op.exists(d):
                    logging.info("Creating [%s]"%d)
                    os.makedirs(d)
        comm.barrier()
    

