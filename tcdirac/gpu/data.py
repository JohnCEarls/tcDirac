import numpy as np
import bisect
import pycuda.driver as cuda
import math

class SharedException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Expression:
    def __init__(self, exp_data=None):
        self.orig_nsamples = None
        self.orig_ngenes = None
        self.orig_data = None

        self.buffer_nsamples = None
        self.buffer_ngenes = None
        self.buffer_data = None

        self.gpu_data = None
        if exp_data is not None:
            self.setData(exp_data)

        
    def gpu_mem(self, sample_block_size, dtype=np.float32):
        if self.buffer_data is not None:
            return self.buffer_data.nbytes
        else:
            bsamp = int(math.ceil(float(self.orig_nsamples)/sample_block_size))*sample_block_size
            return bsamp*dtype(1).nbytes*self.orig_ngenes

    def setData(self, np_data):
        self.orig_nsamples = np_data.shape[1]
        self.orig_ngenes = np_data.shape[0]
        self.orig_data = np_data

    
    def createBuffer(self, sample_block_size, buff_dtype=np.float32):
        """
        create buffer with sample sizes appropriately adjusted
        """

        ns = self.orig_nsamples
        """
        nblocks = ns/sample_block_size 
        if ns%sample_block_size: nblocks += 1"""
        self.buffer_ngenes = self.orig_ngenes
        self.buffer_nsamples =int(math.ceil(float(ns)/sample_block_size))*sample_block_size

        d = self.buffer_data = np.zeros((self.buffer_ngenes, self.buffer_nsamples), dtype=buff_dtype)
        self.buffer_data[:, :ns] = self.orig_data[:,:]

    def toGPU(self, sample_block_size, dtype=np.float32):
        if self.buffer_data is None:
            self.createBuffer( sample_block_size, dtype)
        self.gpu_data = cuda.mem_alloc(self.buffer_data.nbytes)
        cuda.memcpy_htod(self.gpu_data, self.buffer_data)

class SharedExpression(Expression):
    """
    Expression object for shared data
    """    
    def setData(self, np_data):
        self.buffer_nsamples = np_data.shape[1]
        self.buffer_ngenes = np_data.shape[0]
        self.buffer_data = np_data

    
    def createBuffer(self, sample_block_size, buff_dtype=np.float32):
        raise SharedException("cannot create buffer in SharedExpression")

    def toGPU(self, sample_block_size, dtype=np.float):
        self.gpu_data = cuda.mem_alloc(self.buffer_data.nbytes)
        cuda.memcpy_htod(self.gpu_data, self.buffer_data)


class SampleRankTemplate:
    def __init__(self, nsamples, npairs):
        self.res_nsamples = nsamples
        self.res_npairs = npairs
        self.res_data = None

        self.buffer_nsamples = None
        self.buffer_npairs = None
        self.buffer_data = None
        self.buffer_dtype = None
        
        self.gpu_data = None

        
    def gpu_mem(self, sample_block_size, pairs_block_size, dtype=np.int32):
        if self.buffer_data is not None:
            return self.buffer_data.nbytes
        else:

            bns = int(math.ceil(float(self.res_nsamples)/sample_block_size))

            bnp =  int(math.ceil(float(self.res_npairs)/pairs_block_size))
            return  bns*sample_block_size* bnp*pairs_block_size*dtype(1).nbytes



    def toGPU(self, sample_block_size, pairs_block_size, buff_dtype=np.int32):
        self.buffer_dtype = buff_dtype

        #bns = self.res_nsamples/sample_block_size
        #if bns%sample_block_size: bns += 1

        bns = int(math.ceil(float(self.res_nsamples)/sample_block_size))

        #bnp = self.res_npairs/pairs_block_size
        #if bnp%pairs_block_size: bnp += 1
        bnp =  int(math.ceil(float(self.res_npairs)/pairs_block_size))
        
        self.buffer_nsamples = bns*sample_block_size
        self.buffer_npairs = bnp*pairs_block_size

        self.gpu_data = cuda.mem_alloc( self.buffer_nsamples* self.buffer_npairs *buff_dtype(1).nbytes )
        

    def fromGPU(self, res_dtype=np.double):
        if self.gpu_data is not None:
            self.buffer_data = np.empty((self.buffer_npairs,self.buffer_nsamples), dtype = self.buffer_dtype)
            cuda.memcpy_dtoh(self.buffer_data, self.gpu_data)
            self.res_data = np.empty((self.res_npairs, self.res_nsamples), dtype=res_dtype)
            self.res_data[:,:] = self.buffer_data[:self.res_npairs, :self.res_nsamples]

class SharedSampleRankTemplate(SampleRankTemplate):
    def __init__(self, nsamples, npairs):
        self.res_nsamples = None
        self.res_npairs = None
        self.res_data = None

        self.res_nsamples = self.buffer_nsamples = nsamples
        self.res_npairs = self.buffer_npairs = npairs
        self.buffer_data = None
        self.buffer_dtype = None
        
        self.gpu_data = None

        
    def gpu_mem(self, sample_block_size, pairs_block_size, dtype=np.int32):
        bns = self.buffer_nsamples/sample_block_size
        bnp =  self.buffer_npairs/pairs_block_size
        return  bns*sample_block_size* bnp*pairs_block_size*dtype(1).nbytes



    def toGPU(self, sample_block_size, pairs_block_size, buff_dtype=np.int32):
        self.buffer_dtype = buff_dtype
        self.gpu_data = cuda.mem_alloc(self.gpu_mem( sample_block_size, pairs_block_size,buff_dtype ) )
    def fromGPU(self, res_dtype=np.double):
        if self.gpu_data is not None:
            self.buffer_data = np.empty((self.buffer_npairs,self.buffer_nsamples), dtype = self.buffer_dtype)
            cuda.memcpy_dtoh(self.buffer_data, self.gpu_data)
            self.res_data = np.empty((self.res_npairs, self.res_nsamples), dtype=res_dtype)
            self.res_data[:,:] = self.buffer_data[:self.res_npairs, :self.res_nsamples]
      
    """
    def fromGPU(self, res_dtype=np.double):
        raise SharedException("Not Available")"""

class RankTemplate(SampleRankTemplate):
    def __init__(self, nsamples, npairs):
        SampleRankTemplate.__init__(self, nsamples,npairs)

class SharedRankTemplate(SharedSampleRankTemplate):
    def __init__(self, nsamples, npairs):
        SharedSampleRankTemplate.__init__(self, nsamples,npairs)
    
class GeneMap:
    def __init__(self, orig_gmap):
        self.orig_npairs = orig_gmap.shape[0]/2
        self.orig_data = orig_gmap

        self.buffer_npairs = None
        self.buffer_data = None

        self.gpu_data = None

    def gpu_mem(self, pairs_block_size, dtype=np.int32):

        pbs = int(math.ceil(float(self.orig_npairs)/pairs_block_size))
        return 2*pbs*pairs_block_size*dtype(1).nbytes

    def createBuffer(self, pairs_block_size, buff_dtype=np.int32):
        pbs = int(math.ceil(float(self.orig_npairs)/pairs_block_size))
        self.buffer_npairs = pbs*pairs_block_size
        self.buffer_data = np.zeros((self.buffer_npairs*2,), dtype = buff_dtype)
        self.buffer_data[:self.orig_npairs*2] = self.orig_data[:]

    def toGPU(self, pairs_block_size, buff_dtype=np.int32):
        if self.buffer_data is None:
            self.createBuffer( pairs_block_size, buff_dtype)
        self.gpu_data = cuda.mem_alloc( self.buffer_data.nbytes )
        cuda.memcpy_htod(self.gpu_data, self.buffer_data)

    def split(self,partition_point):
        data_1 = self.orig_data[:partition_point*2].copy()
        data_2 = self.orig_data[partition_point*2:].copy()
        gm1 = GeneMap(data_1)
        gm2 = GeneMap(data_2)

        return (gm1,gm2)

class SharedGeneMap(GeneMap):
    def __init__(self, orig_gmap):
        self.orig_npairs = None
        self.orig_data = None

        self.buffer_npairs = orig_gmap.shape[0]/2
        self.buffer_data = orig_gmap

        self.gpu_data = None

    def gpu_mem(self, pairs_block_size, dtype=np.int32):
        return self.buffer_data.nbytes

    def createBuffer(self, pairs_block_size, buff_dtype=np.int32):
        raise SharedException("Not Available")

    def toGPU(self, pairs_block_size, buff_type=np.int32):
        self.gpu_data = cuda.mem_alloc( self.buffer_data.nbytes )
        cuda.memcpy_htod(self.gpu_data, self.buffer_data)

    def split(self, partition_point):
        raise SharedException("No split for shared")
        
class SampleMap:
    def __init__(self, orig_smap):
        self.orig_nsamples = orig_smap.shape[0]
        self.orig_kneighbors = orig_smap.shape[1]
        self.orig_data = orig_smap

        self.buffer_kneighbors = None
        self.buffer_nsamples = None
        self.buffer_data = None
    
        self.gpu_data = None

    def gpu_mem(self, samples_block_size, dtype=np.int32):
        sbs = int(math.ceil(float(self.orig_nsamples)/samples_block_size))

        return sbs*samples_block_size*self.orig_kneighbors*dtype(1).nbytes


    def createBuffer(self, samples_block_size, buff_dtype=np.int32):
        sbs = int(math.ceil(float(self.orig_nsamples)/samples_block_size))
        self.buffer_nsamples = sbs*samples_block_size
        self.buffer_kneighbors = self.orig_kneighbors

        self.buffer_data = np.zeros((self.buffer_nsamples,self.buffer_kneighbors), dtype=buff_dtype)
        self.buffer_data[:self.orig_nsamples, :] = self.orig_data[:,:]

    def toGPU(self,  samples_block_size, buff_dtype=np.int32):
       
        if self.buffer_data is None:
            self.createBuffer( samples_block_size, buff_dtype)
        self.gpu_data = cuda.mem_alloc( self.buffer_data.nbytes )
        cuda.memcpy_htod( self.gpu_data, self.buffer_data )

class SharedSampleMap(SampleMap):
    def __init__(self, orig_smap):
        self.orig_nsamples = None
        self.orig_kneighbors = None
        self.orig_data = None

        self.buffer_kneighbors =  orig_smap.shape[0]
        self.buffer_nsamples =  orig_smap.shape[1]
        self.buffer_data =  orig_smap
    
        self.gpu_data = None

    def gpu_mem(self, samples_block_size, dtype=np.int32):
        return self.buffer_data.nbytes

    def createBuffer(self, samples_block_size, buff_dtype=np.int32):
        raise SharedException("Not Available")

    def toGPU(self,  samples_block_size, buff_dtype=np.int32):
        self.gpu_data = cuda.mem_alloc( self.buffer_data.nbytes )
        cuda.memcpy_htod( self.gpu_data, self.buffer_data )

class SampleMapBin:
    """
    Sample map for binary dirac (i.e. 1-d vector of 1s and 0s)
    """
    def __init__(self, orig_smap):
        raise Exception("Unimplemented")
        self.orig_nsamples = orig_smap.shape[0]
        self.orig_data = orig_smap

        self.buffer_nsamples = None
        self.buffer_data = None

    def gpu_mem(self, samples_block_size, dtype=np.int32):
        sbs = int(math.ceil(float(self.orig_nsamples)/samples_block_size))
        return sbs*samples_block_size*dtype(1).nbytes

    def createBuffer(self, samples_block_size, buff_dtype=np.int32):
        b_size = self.gpu_mem(samples_block_size, buff_dtype)
        self.buffer_nsamples = b_size/buff_dtype(1).nbytes
        self.buffer_data((self.buffer_nsamples,), dtype=buff_dtype)
        self.buffer_data[:self.orig_nsamples] = self.orig_samples[:]
        

    def toGPU(self,  samples_block_size, buff_dtype=np.int32):
       
        if self.buffer_data is None:
            self.createBuffer( samples_block_size, buff_dtype)
        self.gpu_data = cuda.mem_alloc( self.buffer_data.nbytes )
        cuda.memcpy_htod( self.gpu_data, self.buffer_data )

class SharedSampleMapBin(SampleMapBin):
    def toGPU(self,  samples_block_size, buff_dtype=np.int32):
        self.gpu_data = cuda.mem_alloc( self.orig_data.nbytes )
        cuda.memcpy_htod( self.gpu_data, self.orig_data )

    
    
class NetworkMap:
    """
    1-d array with offsets for each network
    i.e. [0,size(net_0),sum(size(net_i),0,i) ... sum(size_(net_i),0,n)]
    """
    def __init__(self, orig_net_map):
        self.orig_data = orig_net_map
        self.orig_nnets = len(orig_net_map) - 1

        self.buffer_nnets = None
        self.buffer_data = None
       
        self.gpu_data = None

    def gpu_mem(self, nnets_block_size, dtype=np.int32):
        nbs = int(math.ceil(float(self.orig_nnets)/nnets_block_size))
        return nbs*(nnets_block_size+1)*dtype(1).nbytes


    def createBuffer(self, nnets_block_size, buff_dtype=np.int32):
        nbs = int(math.ceil(float(self.orig_nnets)/nnets_block_size))
        self.buffer_nnets = nbs*nnets_block_size
        self.buffer_data = np.zeros( (self.buffer_nnets+1,), dtype=buff_dtype)
        self.buffer_data[:len(self.orig_data)] = self.orig_data[:]


        
    def toGPU(self, nets_block_size, buff_dtype = np.int32):
        if self.buffer_data is None:
            self.createBuffer(nets_block_size, buff_dtype)
        self.gpu_data = cuda.mem_alloc( self.buffer_data.nbytes )
        cuda.memcpy_htod(self.gpu_data, self.buffer_data)
    def split(self):
        
        center = bisect.bisect(self.orig_data, self.orig_data[-1]/2)
        if center == len(self.orig_data) - 1:
            center -= 1
        assert 0 < center < len(self.orig_data), "If this assert is reached, then a single network is too big for memory.If you want to fix this you will have to rework the code"
        if center == len(self.orig_data) - 1:
            center -= 1
        part = self.orig_data[center]
        new_1 = self.orig_data[:center+1].copy()
        new_2 = self.orig_data[center:].copy()
        new_2 =  new_2 - part

        nm1 = NetworkMap(new_1)
        nm2 = NetworkMap(new_2)

        return (part, nm1, nm2)

class SharedNetworkMap(NetworkMap):

    def __init__(self, orig_net_map):

        self.orig_data = None
        self.orig_nnets = self._get_orig_nnets(orig_net_map) 

        self.buffer_data = orig_net_map
        self.buffer_nnets = len(orig_net_map) - 1
       
        self.gpu_data = None

    def _get_orig_nnets(self, orig_net_map):
        ctr = 1
        lo = len(orig_net_map)
        assert lo - ctr > 1, "%s %s"%(str(lo),str(ctr))
            
        while orig_net_map[lo - ctr]==0:
            ctr += 1
            assert lo - ctr > 1, "%s %s %s"%(str(lo),str(ctr),str(orig_net_map))
        return lo - ctr

    def gpu_mem(self, nnets_block_size, dtype=np.int32):
        return self.buffer_data.nbytes
    

    def createBuffer(self, nnets_block_size, buff_dtype=np.int32):
        raise SharedException("Not Available")

    def split(self):
        raise SharedException("No splits for Shared memory")

    def toGPU(self, nets_block_size, buff_dtype = np.int32):
        self.gpu_data = cuda.mem_alloc( self.buffer_data.nbytes )
        cuda.memcpy_htod(self.gpu_data, self.buffer_data)

class RankMatchingScores:
    def __init__(self, num_nets, nsamples):
        self.res_data = None
        self.res_nnets = num_nets
        self.res_nsamples = nsamples

        self.buffer_nnets = None
        self.buffer_nsamples = None
        self.buffer_data = None

        self.gpu_data = None

    def gpu_mem(self, samples_block_size, nets_block_size, dtype=np.float32):
        a = int(math.ceil(float(self.res_nsamples)/samples_block_size))* samples_block_size
        b = int(math.ceil(float(self.res_nnets)/nets_block_size))*nets_block_size
        return a*b*dtype(1).nbytes

    def createBuffer(self, samples_block_size, nets_block_size, buff_dtype=np.float32):
        self.buffer_nsamples = int(math.ceil(float(self.res_nsamples)/samples_block_size))* samples_block_size
        self.buffer_nnets = int(math.ceil(float(self.res_nnets)/nets_block_size))*nets_block_size
        self.buffer_data = np.zeros( (self.buffer_nnets, self.buffer_nsamples), dtype=buff_dtype)

   
    def toGPU(self,  samples_block_size, nets_block_size, buff_dtype=np.float32):
        if self.buffer_data is None:
            self.createBuffer( samples_block_size, nets_block_size, buff_dtype=np.float32)
        self.gpu_data = cuda.mem_alloc( self.buffer_data.nbytes )


    def fromGPU(self, res_dtype=np.double):
        if self.gpu_data is not None:
            cuda.memcpy_dtoh(self.buffer_data, self.gpu_data)
            self.res_data = np.empty(( self.res_nnets, self.res_nsamples), dtype = res_dtype)
            self.res_data[:,:] = self.buffer_data[:self.res_nnets, :self.res_nsamples]

class SharedRankMatchingScores(RankMatchingScores):
    """
    RMS written to shared memory
    """
    def __init__(self, num_nets, nsamples):
        self.res_data = None
        self.res_nnets = None
        self.res_nsamples = None

        self.buffer_nnets = num_nets
        self.buffer_nsamples = nsamples
        self.buffer_data = None

        self.gpu_data = None

    def toGPU(self, samples_block_size, nets_block_size, buff_dtype=np.float32):
        self.gpu_data = cuda.mem_alloc( self.gpu_mem( samples_block_size, nets_block_size, buff_dtype) )

    def fromGPU(self, shared_mem, buff_dtype=np.float32 ):
        
        buff = np.frombuffer(shared_mem.get_obj(), dtype=buff_dtype)
        buff = buff[:self.buffer_nnets*self.buffer_nsamples]
        buff = buff.reshape( (self.buffer_nnets, self.buffer_nsamples) )
        """
        self.res_data = np.zeros(( self.buffer_nnets, self.buffer_nsamples), dtype = buff_dtype)
        cuda.memcpy_dtoh(self.res_data, self.gpu_data )
        """
        cuda.memcpy_dtoh(buff, self.gpu_data)
        return buff
        #print "All close", np.allclose( buff, self.res_data)

    def gpu_mem(self, samples_block_size, nets_block_size, dtype=np.float32):
        return self.buffer_nnets*self.buffer_nsamples*dtype(1).nbytes

    def createBuffer(self, samples_block_size, nets_block_size, buff_dtype=np.float32):
        raise SharedException("Not Available")
        
if __name__ == "__main__":
    import pandas

    cuda.init()
    n = 200
    gn = 1000
    nets = 5 
    sample_block_size = 32
    npairs_block_size = 16
    nets_block_size = 8
    for n in range(100,2000,100):
        dev = cuda.Device(1)
        ctx = dev.make_context()
        init_free, total = cuda.mem_get_info()
        b2_size = 32
        genes = map(lambda x:'g%i'%x, range(gn))
        samples = map(lambda x:'s%i'%x, range(n))
        exp = pandas.DataFrame(np.random.rand(len(genes),len(samples)), index=genes, columns=samples)

        
        pred_used = 0
        e = Expression( exp.values )
        pred_used += e.gpu_mem(sample_block_size, dtype=np.float32)
        
        e.toGPU(sample_block_size)


        srt = SampleRankTemplate(len(samples),111776 )

        pred_used += srt.gpu_mem( sample_block_size, npairs_block_size, dtype=np.float32)
        srt.toGPU( sample_block_size, npairs_block_size )
        srt.fromGPU()

        rt = RankTemplate(len(samples), 111776)
        rt.toGPU( sample_block_size, npairs_block_size )
        rt.fromGPU()
        pred_used += rt.gpu_mem( sample_block_size, npairs_block_size, dtype=np.int32)

        nm_orig = np.array(range(0, 200,1))
        
        nm = NetworkMap(nm_orig )
        nm.toGPU( nets_block_size )
        pred_used += nm.gpu_mem( nets_block_size )

        
        gm = GeneMap( np.array(range(1000)))
        gm.toGPU(nets_block_size )

        pred_used += nm.gpu_mem( nets_block_size )

        s_map = np.random.randint(low=0,high=len(samples), size=(len(samples),5 )).astype(np.int32)
        for i in range(s_map.shape[0]):
            s_map[i,0] = i
        sm = SampleMap(s_map)
        sm.toGPU( sample_block_size )
        pred_used += sm.gpu_mem( sample_block_size )

        final_free, total = cuda.mem_get_info() 
        print "actual",init_free - final_free
        print "pred",pred_used
        print "ratio",float(pred_used)/ (init_free - final_free)


        ctx.pop()

        
