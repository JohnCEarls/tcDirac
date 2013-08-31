
import pycuda.driver as cuda
#import pycuda.autoinit
import atexit
from pycuda.compiler import SourceModule
import numpy as np
import math
from mpi4py import MPI
class Kern:
    dirac = """

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



class Dirac:

    def __init__(self, expression_matrix, device_id=0):
        
        self.exp = expression_matrix
        self.exp_gpu = None
        self.srtKern = None
        #pycuda.autoinit.context.get_device(device_id) 
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
            return srt_gpu
        else:
            srt_buffer = np.zeros(srt_shape, dtype=np.int32)
            cuda.memcpy_dtoh(srt_buffer, srt_gpu)
            srt_gpu.free()

            return srt_buffer[:gmap.shape[0]/2,:self.exp.shape[1]]
            

        

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
            mod = SourceModule(Kern.dirac)
            func = mod.get_function("srtKernel")
            self.srtKern = func
        return self.srtKern
        
        
    def clearExp(self):
        if self.exp_gpu is not None:
            self.exp_gpu.free()
            self.exp_gpu = None

        


    def getGrid(self,num_items, b_size=32):
        g = int(num_items/b_size)#get grid spacing
        if num_items%b_size != 0:
            g += 1
        return g


def getTestExp(ngenes,nsamples):
    exp = np.zeros((ngenes,nsamples))
    for i in range(exp.shape[0]):
        for j in range(exp.shape[1]):
            exp[i,j] = math.pow(-1,j)*(  i*1000 + j)

    return exp

def getGmap(netsize):
    tmp_gm = []
    for i in range(netsize-1):
        for j in range(i+1,netsize):
            tmp_gm += [i,j]

    return np.array(tmp_gm)
if __name__ == "__main__":
    comm =MPI.COMM_WORLD
    
    
    exp = getTestExp(ngenes=200, nsamples=500)
    gm = getGmap(100) 
    d = Dirac(exp, device_id=comm.rank%2)
    #d = Dirac(exp)
    d.initExp()
    #att = cuda.Context.get_device().get_attributes()
    """
    for k,v in att.iteritems():
        if str(k) == 'PCI_DEVICE_ID':
            print comm.rank, v"""
    for _ in range(160):
        for i in range(10, 20):
            gm = getGmap(10*i)
            srt =  d.getSRT(gm)
                
        #print srt.shape
        #print srt
    

