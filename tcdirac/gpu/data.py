import numpy as np
import pycuda.driver as cuda
import math
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

        

    def setData(self, np_data):
        self.orig_nsamples = np_data.shape[1]
        self.orig_ngenes = np_data.shape[0]
        self.orig_data = np_data

    
    def createBuffer(self, sample_block_size, dtype=np.int32):
        """
        create buffer with sample sizes appropriately adjusted
        """

        ns = self.orig_nsamples
        nblocks = ns/sample_block_size 
        if ns%sample_block_size: nblocks += 1
        self.buffer_ngenes = self.orig_ngenes
        self.buffer_nsamples = nblocks * sample_block_size

        d = self.buffer_data = np.zeros((self.buffer_ngenes, self.buffer_nsamples), dtype=np.int32)
        d[:, :ns] = self.orig_data[:,:]

    def toGPU(self, sample_block_size, dtype=np.int32):
        if self.buffer_data is None:
            self.createBuffer( sample_block_size, dtype)
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

    def toGPU(self, sample_block_size, pairs_block_size, buff_dtype=np.int32):
        self.buffer_dtype = buff_dtype

        bns = self.res_nsamples/sample_block_size
        if bns%sample_block_size: bns += 1

        bnp = self.res_npairs/pairs_block_size
        if bnp%pairs_block_size: bnp += 1
        
        self.buffer_nsamples = bns*sample_block_size
        self.buffer_npairs = bnp*pairs_block_size

        self.gpu_data = cuda.mem_alloc( self.buffer_nsamples* self.buffer_npairs *buff_dtype(1).nbytes )
        

    def fromGPU(self, res_dtype=np.double):
        self.buffer_data = np.empty((self.buffer_npairs,self.buffer_nsamples), dtype = self.buffer_dtype)
        cuda.memcpy_dtoh(self.buffer_data, self.gpu_data)
        self.res_data = np.empty((self.res_npairs, self.res_nsamples), dtype=res_dtype)
        self.res_data[:,:] = self.buffer_data[:self.res_npairs, :self.res_nsamples]

class RankTemplate(SampleRankTemplate):
    def __init__(self, nsamples, npairs):
        super(SampleRankTemplate, self).__init__(nsamples,npairs)

    
class GeneMap:
    def __init__(self, orig_gmap):
        self.orig_npairs = orig_gmap.shape[0]/2
        self.orig_data = orig_gmap

        self.buffer_npairs = None
        self.buffer_data = None

        self.gpu_data = None

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

        
class SampleMap:
    def __init__(self, orig_smap):
        self.orig_nsamples = orig_smap.shape[0]
        self.orig_kneighbors = orig_smap.shape[1]
        self.orig_data = orig_smap

        self.buffer_kneighbors = None
        self.buffer_nsamples = None
        self.buffer_data = None

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

    def createBuffer(self, nnets_block_size, buff_dtype=np.int32)
        nbs = int(math.ceil(float(self.orig_nnets)/nnets_block_size))
        self.buffer_nnets = nbs*nnets_block_size
        self.buffer_data = np.zeros( (self.buffer_nnets+1,) dtype=buff_dtype)

        self.buffer_data[:len(orig_data)] = self.orig_data[:]

        
    def toGPU(self, nets_block_size, buff_dtype = np.int32):
        if self.buffer_data is None:
            self.createBuffer(nets_block_size, buff_dtype)
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

    def createBuffer(self, samples_block_size, nets_block_size, buff_dtype=np.float32):
        self.buffer_nsamples = int(math.ceil(float(self.nsamples)/samples_block_size))* samples_block_size
        self.buffer_nnets = int(math.ceil(float(res_nnets)/nets_block_size)*nets_block_size
        self.buffer_data = np.zeros( (self.buffer_nnets, self.buffer_nsamples), dtype=buff_dtype)

   
    def toGPU(self,  samples_block_size, nets_block_size, buff_dtype=np.float32):
        if self.buffer_data is None:
            self.createBuffer( samples_block_size, nets_block_size, buff_dtype=np.float32)
        self.gpu_data = cuda.mem_alloc( self.buffer_data.nbyte )


    def fromGPU(self):
        cuda.memcpy_dtoh(self.buffer_data, self.gpu_data)
        self.res_data[:,:] = self.buffer_data[:self.res_nnets, :self.res_nsamples]


        
if __name__ == "__main__":
    import pandas

    cuda.init()
    n = 200
    gn = 1000
    b2_size = 32
    genes = map(lambda x:'g%i'%x, range(gn))
    samples = map(lambda x:'s%i'%x, range(n))
    exp = pandas.DataFrame(np.random.rand(len(genes),len(samples)), index=genes, columns=samples)

    dev = cuda.Device(0)
    ctx = dev.make_context()

    e = Expression( exp.values )
    e.toGPU(b2_size)

    assert e.buffer_nsamples%b2_size == 0

    srt = SampleRankTemplate(len(samples), 200)
    srt.toGPU( 32, 16)
    srt.fromGPU()
    print srt.res_data

    gm = GeneMap( np.array(range(1000)))
    gm.toGPU(16)
    print gm.buffer_data[:20]

    s_map = np.random.randint(low=0,high=len(samples), size=(len(samples),5 )).astype(np.int32)
    for i in range(s_map.shape[0]):
        s_map[i,0] = i
    sm = SampleMap(s_map)
    sm.toGPU(32)
    print sm.buffer_data


    ctx.pop()

        
