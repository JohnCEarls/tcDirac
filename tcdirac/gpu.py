import pandas
import random
import pycuda.driver as cuda
#import pycuda.autoinit
import atexit
from pycuda.compiler import SourceModule
import numpy as np
import math
from mpi4py import MPI
import itertools
import time
import pycuda.gpuarray as ga
class Kern:
    srt = """

    __global__ void srtKernel( float * d_expression, int nsamples, int ngenes, int *gene_map, int npairs, int * srt){
        int g_pair = blockIdx.x*blockDim.x + threadIdx.x;//(0,npairs-1)
        int sample = blockIdx.y*blockDim.y + threadIdx.y;//(0,nsamples-1)
       
        int gm1 = gene_map[ 2*g_pair ];
        int gm2 = gene_map[ 2*g_pair + 1 ];
        
        
        float gene_1_exp = d_expression[gm1*nsamples + sample];

        float gene_2_exp = d_expression[gm2*nsamples + sample];
        srt[nsamples*g_pair + sample] = (int)(gene_1_exp < gene_2_exp);
    }


    """
    rt1 = """
    __global__ void rtKernel1( int * srt, int nsamples, int gpairs, int * sample_map, int * rt,int num_blocks){
        extern __shared__ int s_srt[];
        
    
        int g_pair = blockIdx.x*blockDim.x + threadIdx.x;//(0,npairs-1)
        int sample =  blockIdx.y*blockDim.y + threadIdx.y;

        int block_g_pair = threadIdx.x;
        int block_sample = threadIdx.y;
        int block_nsamples = blockDim.y;
        
        s_srt[block_nsamples*block_g_pair + block_sample] = srt[nsamples*g_pair + sample] & sample_map[sample];
       
        
        for(unsigned int stride = blockDim.y / 2; stride > 0; stride /= 2){
            __syncthreads();
            if( block_sample < stride){
                s_srt[block_nsamples*block_g_pair + block_sample] += s_srt[block_nsamples*block_g_pair + block_sample + stride];
            }
        }
        __syncthreads();
        if( block_sample == 0){
                rt[num_blocks*g_pair + blockIdx.y] = s_srt[block_nsamples*block_g_pair];
        }
    }    

    

"""

    rms = """
        __global__ void rmsKernel( int * rt, int * srt, int sample_id, int padded_samples,  int true_npairs, int * result){
            extern __shared__ int s_srt[];
            int i = blockIdx.x*blockDim.x + threadIdx.x;
            s_srt[threadIdx.x] = ((rt[i] == srt[i * padded_samples + sample_id]) && (i < true_npairs));
            
            for(unsigned int stride= blockDim.x/2; stride > 0; stride/=2){
                __syncthreads();
                if (threadIdx.x < stride)
                    s_srt[threadIdx.x] += s_srt[threadIdx.x + stride];
            }
            if (threadIdx.x == 0)
                result[blockIdx.x] = s_srt[0];
        }


    """

def rtfinish(nblocks):
    a =  """
__global__ void rtKernel2( int * toReduce, int * final, int nsamples){
    int g_pair = blockIdx.x*blockDim.x + threadIdx.x;
    int count = %s;
    final[g_pair] = nsamples/2 < count;
}"""% '+'.join(['toReduce[%i*g_pair + %i]'%(nblocks,i) for i in range(nblocks)])
    return a
class Dirac:

    def __init__(self, expression_matrix, device_id=0):
        
        self.exp = expression_matrix
        self.exp_gpu = None
        self.srtKern = None
        self.rt1Kern = None
        self.rt2Kern = {}
        self.rmsKern = None
        drv = cuda
        drv.init()
        dev = drv.Device(device_id)
        ctx = dev.make_context()
        atexit.register(ctx.pop)
    

    def getBuff(self, frm, new_r, new_c, b_dtype):
        """
        Generates a numpy array sized (new_r,new_x) of dtype
            b_dtype that contains the np array frm such that
            frm[i,j] == new[i,j] wher new has zeros if
            frm[i,j] is out of bounds.
        """
        try:
            old_r,old_c =  frm.shape
            buff = np.zeros((new_r,new_c),dtype=b_dtype)
            buff[:old_r,:old_c] = frm
        except ValueError:
            #oned
            old_r = frm.shape[0]
            buff = np.zeros((new_r,),dtype=b_dtype)
            buff[:old_r] = frm
        return buff

    def getSRT(self, gmap, b_size=32, store_srt=False):
        """
        exp is a 2d numpy array shaped genesxsamples
            i.e exp[0,3] is gene 0 for sample 3
        gmap is a 1d numpy array where gmap[2*i] and gmap[2*i +1] are 
            gene indices for comparison i
        """
        #the x coords in the gpu map to sample_ids
        #the y coords to gmap
        #sample blocks
        exp = self.exp 
        g_y_sz = self.getGrid( exp.shape[1] )
        #pair blocks
        g_x_sz = self.getGrid( gmap.shape[0]/2 )

        #put gene map on gpu
        gmap_buffer = self.getBuff(gmap, 2*(g_x_sz*b_size), 1,np.int32)
        gmap_gpu = cuda.mem_alloc(gmap_buffer.nbytes)
        cuda.memcpy_htod(gmap_gpu,gmap_buffer)

        #make room for srt
        srt_shape = (g_x_sz*b_size , g_y_sz*b_size)
        srt_gpu = cuda.mem_alloc(srt_shape[0]*srt_shape[1]*np.int32(1).nbytes)

        srtKern = self.getsrtKern()
        

        exp_gpu = self.exp_gpu
        nsamp = np.uint32( g_y_sz * b_size )
        ngenes = np.uint32( self.exp.shape[0] )
        npairs = np.uint32( g_x_sz * b_size )

        block = (b_size,b_size,1)
        grid = (g_x_sz, g_y_sz)

        srtKern(exp_gpu, nsamp, ngenes, gmap_gpu, npairs, srt_gpu, block=block, grid=grid)

        gmap_gpu.free()
        
       
        if store_srt:
            #this is in case we want to run further stuff without 
            #transferring back and forth
            return (srt_gpu, npairs , nsamp)
        else:
            srt_buffer = np.zeros(srt_shape, dtype=np.int32)
            cuda.memcpy_dtoh(srt_buffer, srt_gpu)
            srt_gpu.free()

            return srt_buffer[:gmap.shape[0]/2,:self.exp.shape[1]]
            

    def getRT(self, s_map, srt_gpu, srt_nsamp, srt_npairs, npairs, b_size=32, store_rt=False):

        s_map_buff = np.array(s_map + [0 for i in range(srt_nsamp - len(s_map))],dtype=np.int32)

        s_map_gpu = cuda.mem_alloc(s_map_buff.nbytes)
        cuda.memcpy_htod(s_map_gpu, s_map_buff)
        
        #sample blocks
        g_y_sz = self.getGrid( srt_nsamp, b_size )
        #pair blocks
        g_x_sz = self.getGrid( srt_npairs,b_size )
        
        block_rt_gpu = cuda.mem_alloc(int(g_y_sz*srt_npairs*(np.uint32(1).nbytes)) ) 
        rt_gpu = cuda.mem_alloc(int(srt_npairs*(np.uint32(1).nbytes))) 

        grid = (g_x_sz, g_y_sz)

        func1,func2 = self.getrtKern(g_y_sz)

        shared_size = b_size*b_size*np.uint32(1).nbytes
        #int * srt, int nsamples, int gpairs, int * sample_map, int * rt,int num_blocks

        func1( srt_gpu, np.uint32(srt_nsamp), np.uint32(srt_npairs), s_map_gpu, block_rt_gpu, np.uint32(g_y_sz), block=(b_size,b_size,1), grid=grid, shared=shared_size)


        func2( block_rt_gpu, rt_gpu, np.int32(s_map_buff.sum()), block=(b_size,1,1), grid=(g_x_sz,))

        
        if store_rt:
            #this is in case we want to run further stuff without 
            #transferring back and forth
            return (rt_gpu, srt_npairs)
        else:
            rt_buffer = np.zeros((srt_npairs ,), dtype=np.int32)
            cuda.memcpy_dtoh(rt_buffer, rt_gpu)
            rt_gpu.free()
            return rt_buffer[:npairs]

    def getRMS(self, rt_gpu, srt_gpu, padded_samples, padded_npairs, samp_id, npairs, b_size=32):
        gsize = int(padded_npairs/b_size)
        result = np.zeros((gsize,), dtype=np.int32)
        result_gpu = cuda.mem_alloc(result.nbytes)
         
        #result = np.empty((gsize,), dtype=np.int32)
        func = self.getrmsKern()
        func( rt_gpu, srt_gpu, np.int32(samp_id), np.int32(padded_samples), np.int32(npairs), result_gpu, block=(b_size,1,1), grid=(int(gsize),), shared=b_size*np.uint32(1).nbytes )
        #result = np.zeros((gsize,), dtype=np.int32)
        cuda.memcpy_dtoh(result, result_gpu)
        return result.sum()/float(npairs)

        #__global__ void rmsKernel( int * rt, int * srt, int sample_id, int padded_samples  int true_npairs, int * result){


    def initExp(self, b_size=32):
        """
        pads the expression matrix to fit in gpu
        sends the padded expression matrix to gpu
        """ 
        self.clearExp()
        exp = self.exp
        g_x_sz = self.getGrid( exp.shape[1], b_size )
        exp_buffer = self.getBuff(exp,  exp.shape[0], g_x_sz*b_size, np.float32)
        
        exp_gpu = cuda.mem_alloc(exp_buffer.nbytes)
        cuda.memcpy_htod(exp_gpu, exp_buffer)
        self.exp_gpu = exp_gpu


    def getsrtKern(self):
        if self.srtKern is None:
            mod = SourceModule(Kern.srt)
            func = mod.get_function("srtKernel")
            self.srtKern = func
        return self.srtKern

    def getrtKern(self,nblocks):
        if self.rt1Kern is None:
            mod = SourceModule(Kern.rt1)
            func1 = mod.get_function("rtKernel1")
            self.rt1Kern = func1

        if nblocks not in self.rt2Kern:
            mod = SourceModule(rtfinish(nblocks))
            func2 = mod.get_function("rtKernel2")
            self.rt2Kern[nblocks] = func2
        return (self.rt1Kern, self.rt2Kern[nblocks])

    def getrmsKern(self):
        if self.rmsKern is None:
            mod = SourceModule(Kern.rms)
            func = mod.get_function("rmsKernel")
            self.rmsKern = func
        return self.rmsKern
        
        
    def clearExp(self):
        if self.exp_gpu is not None:
            self.exp_gpu.free()
            self.exp_gpu = None

        


    def getGrid(self,num_items, b_size=32):
        g = int(num_items/b_size)#get grid spacing
        if num_items%b_size != 0:
            g += 1
        return g


def testSRT():
    
    for n in range(100,120,17):
        samples = map(lambda x:'s%i'%x, range(n))
        for gn in range(1000,1111,371):
            suck = False
            genes = map(lambda x:'g%i'%x, range(gn))

            exp = pandas.DataFrame(np.zeros((len(genes),len(samples)), dtype=float), index=genes, columns=samples)
            
            for i,g in enumerate(genes):
                for j,s in enumerate(samples):
                    v = random.random()
                    exp.loc[g,s] = v

            np_exp = exp.values
            genes = exp.index
            g_d = {}
            for i,gene in enumerate(genes):
                g_d[gene] = i 
            samples = exp.columns
            d = Dirac(np_exp)
            print "a"
            d.initExp()
            print "b" 
            for ns in range(10, 100,17):
                ic = itertools.combinations
                gm_list = []
                net = random.sample(genes,ns) 
                for g1,g2 in ic(net,2):
                    gm_list += [g_d[g1],g_d[g2]]
                gm = np.array(gm_list, dtype=np.int32)
                srt = d.getSRT(gm)
                res = pandas.DataFrame( srt,index=["%s < %s" % (g1,g2) for g1, g2 in  ic(net,2)],columns=samples)
                """
                for i in res.index:
                    g1,g2 = i.split(' < ')
                    
                    for s in samples:
                        #if ns == 10 and not suck:
                        #    print "here"
                        #    suck = True
                        if res[s][i] == 1:
                            assert exp[s][g1] < exp[s][g2], "does not match"
                        else:
                            assert exp[s][g1] >= exp[s][g2] or np.allclose([exp[s][g1]], [exp[s][g2]], .01) , "does not match[%f] [%f] [%s][%s]" % (exp[s][g1], exp[s][g2], g1, g2)"""
                
                
    
def testRT():
    n=10
    gn = 3000
    ns = 10
    for n in range(10,100,13):

        samples = map(lambda x:'s%i'%x, range(n))
        genes = map(lambda x:'g%i'%x, range(gn))

        exp = pandas.DataFrame(np.random.rand(len(genes),len(samples)), dtype=float, index=genes, columns=samples)
        np_exp = exp.values
        genes = exp.index
        g_d = {}
        for i,gene in enumerate(genes):
            g_d[gene] = i 
        samples = exp.columns
        d = Dirac(np_exp)
        d.initExp()

        for ns in range(10,200,17):   
            print "testRT: num_samples[%i] net_size[%i]" % ( n, ns )
            test_start = time.time()
            ic = itertools.combinations
            gm_list = []
            net = random.sample(genes,ns) 
            for g1,g2 in ic(net,2):
                gm_list += [g_d[g1],g_d[g2]]
            gm = np.array(gm_list, dtype=np.int32)

            srt_start = time.time()
            (srt_gpu, srt_npairs, srt_nsamp) = d.getSRT(gm,store_srt=True)
            srt_end = time.time()
            print "srt [%f]" % (srt_end-srt_start)
            s_map = []
            for i in range(n):
                if random.random() > .8:
                    s_map.append(1)
                else:
                    s_map.append(0)

            #print s_map
            rt_start = time.time()
            rt = d.getRT(s_map, srt_gpu, srt_nsamp, srt_npairs, len(gm_list)/2) 
            rt_end = time.time()

            print "rt [%f]" % (rt_end-rt_start)
            #print rt
            srt_gpu.free()
            res = pandas.Series( rt,index=["%s < %s" % (g1,g2) for g1, g2 in  ic(net,2)])
            #print res
            ctr = 0
            for g1, g2 in  ic(net,2):
                if True or ctr < 1:
                    mysum = 0
                    for s,i in zip(exp.columns, s_map):
                        if exp[s][g1] < exp[s][g2]:
                            mysum += i 
                    if sum(s_map)/2 < mysum:
                        assert(res["%s < %s" % (g1,g2)] == 1)
                    else:
                        assert(res["%s < %s" % (g1,g2)] == 0)
                    
                ctr += 1
            test_end = time.time()
            print "test [%f]" % (test_end-test_start)
        


def testRMS():

    n=10
    gn = 3000
    ns = 10
    for n in [100]:#range(10,100,13):

        samples = map(lambda x:'s%i'%x, range(n))
        genes = map(lambda x:'g%i'%x, range(gn))

        exp = pandas.DataFrame(np.random.rand(len(genes),len(samples)), dtype=float, index=genes, columns=samples)
        np_exp = exp.values
        genes = exp.index
        g_d = {}
        for i,gene in enumerate(genes):
            g_d[gene] = i 
        samples = exp.columns
        d = Dirac(np_exp)
        d.initExp()

        for ns in [50]:#range(10,200,17):   
            print "testRT: num_samples[%i] net_size[%i]" % ( n, ns )
            test_start = time.time()
            ic = itertools.combinations
            gm_list = []
            net = random.sample(genes,ns) 
            for g1,g2 in ic(net,2):
                gm_list += [g_d[g1],g_d[g2]]
            gm = np.array(gm_list, dtype=np.int32)

            srt_start = time.time()
            (srt_gpu, srt_npairs, srt_nsamp) = d.getSRT(gm,store_srt=True)
            srt_end = time.time()
            print "srt [%f]" % (srt_end-srt_start)
            s_map = []
            for i in range(n):
                if random.random() > .8:
                    s_map.append(1)
                else:
                    s_map.append(0)

            #print s_map
            rt_start = time.time()
            rt_gpu, rt_len  = d.getRT(s_map, srt_gpu, srt_nsamp, srt_npairs, len(gm_list)/2, store_rt=True) 
            rt_end = time.time()

            print "rt [%f]" % (rt_end-rt_start)
            #print rt
            #srt_gpu.free()
            """
            res = pandas.Series( rt,index=["%s < %s" % (g1,g2) for g1, g2 in  ic(net,2)])
            #print res
            ctr = 0
            for g1, g2 in  ic(net,2):
                if True or ctr < 1:
                    mysum = 0
                    for s,i in zip(exp.columns, s_map):
                        if exp[s][g1] < exp[s][g2]:
                            mysum += i 
                    if sum(s_map)/2 < mysum:
                        assert(res["%s < %s" % (g1,g2)] == 1)
                    else:
                        assert(res["%s < %s" % (g1,g2)] == 0)
                    
                ctr += 1"""
            test_end = time.time()
            print "test [%f]" % (test_end-test_start)
            for i in range(n):
                rms = d.getRMS(rt_gpu, srt_gpu, srt_nsamp, srt_npairs, i, len(gm_list)/2)    
                print rms

            
def getGmap(netsize):
    tmp_gm = []
    for i in range(netsize-1):
        for j in range(i+1,netsize):
            tmp_gm += [i,j]

    return np.array(tmp_gm)


def getBuff(frm,new_r, new_c,b_dtype):
    try:
        old_r,old_c =  frm.shape
        buff = np.zeros((new_r,new_c),dtype=b_dtype)
        buff[:old_r,:old_c] = frm
    except ValueError:
        #oned
        old_r = frm.shape[0]
        buff = np.zeros((new_r,),dtype=b_dtype)
        buff[:old_r] = frm
    return buff
    
if __name__ == "__main__":
    testRMS()
    """
    print rtfinish(3)

    b_size = 32 


    F_BYTES = np.float32(1.0).nbytes
    I_BYTES = np.int32(1).nbytes

    ngenes = 20 
    npairs = ngenes 
    nsamples = 64 
    x = np.float32(1.0)
    exp = np.zeros((ngenes,nsamples))
    for i in range(exp.shape[0]):

            exp[i,j] = i%2 


    tmp_gm = [i%2 for i in range(nsamples)]
    ctr = 0


    s_map = np.array(tmp_gm)
    

    #npairs (x axis on gpu)
    g1 = int(npairs/b_size)#get grid spacing


    if npairs%b_size != 0:
        g1 += 1

    #nsmples (y axis on gpu)

    g2 = int(nsamples/b_size)#get grid spacing

    #pairs x samples
    if nsamples%b_size != 0:
        g2 += 1

    print "g1(gp)",g1
    print "g2(samp)", g2

    npairs_buff = g1*b_size
    nsamp_buff = g2*b_size

    exp_buffer = getBuff(exp, ngenes , g2*b_size, np.int32)
    s_map_buffer = getBuff(s_map,(g2*b_size), 1,np.int32)
    srt_buffer_size = (g2*(g1*b_size))*I_BYTES

    exp_gpu = cuda.mem_alloc(exp_buffer.nbytes)
    s_map_gpu = cuda.mem_alloc(s_map_buffer.nbytes)
    srt_gpu = cuda.mem_alloc(srt_buffer_size)
    rt_gpu = cuda.mem_alloc( npairs_buff*I_BYTES )

    cuda.memcpy_htod(exp_gpu, exp_buffer)
    cuda.memcpy_htod(s_map_gpu, s_map_buffer)

    mod = SourceModule(Kern.rt)
    func1 = mod.get_function("rtKernel1")

    shared_size = b_size*b_size*I_BYTES

    func1( exp_gpu, np.uint32(nsamp_buff), np.uint32(npairs_buff), s_map_gpu, srt_gpu, np.uint32(g2), block=(b_size,b_size,1), grid=(g1,g2), shared=shared_size)
    mod = SourceModule(rtfinish(g2))
    func2 = mod.get_function("rtKernel2")
    func2( srt_gpu, rt_gpu, np.int32(s_map.sum()), block=(b_size,1,1), grid=(g1,))
    
    rt_buff = np.empty( (npairs_buff,),dtype=np.int32)
    cuda.memcpy_dtoh(rt_buff, rt_gpu)
    print rt_buff
    #srt_buff = np.empty( (npairs_buff,g2 ),dtype=np.uint32)
    #cuda.memcpy_dtoh(srt_buff, srt_gpu)
    exp_gpu.free()
    srt_gpu.free()
    s_map_gpu.free()"""
    """
    print "exp"
    print exp_buffer[:4,:].astype(int)

    print "gmap"
    print gmap"""

    #print "srt_buff"
    #srt = srt_buff #[:npairs, :nsamples]
    #print srt
    #print srt.shape
    """
    testSRT()

    comm =MPI.COMM_WORLD
    
    
    exp = getTestExp(ngenes=200, nsamples=500)
    gm = getGmap(100) 
    d = Dirac(exp, device_id=comm.rank%2)
    #d = Dirac(exp)
    d.initExp()
    #att = cuda.Context.get_device().get_attributes()
    
    for k,v in att.iteritems():
        if str(k) == 'PCI_DEVICE_ID':
            print comm.rank, v
    for _ in range(160):
        for i in range(10, 20):
            gm = getGmap(10*i)
            srt =  d.getSRT(gm)
                
        #print srt.shape
        #print srt"""
    

