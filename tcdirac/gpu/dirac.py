debug = True
int32 = 4
import kernels
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
def sampleRankTemplate( exp_gpu, gmap_gpu, nsamp, npairs, b1_size, b2_size ):
    """
    nsamp  is the columns dim (shape[1]) of exp_gpu
    npairs is the length of gmap_gpu (shape[0])
    """

    if debug:
        assert npairs%b1_size == 0
        assert nsamp%b2_size == 0
    
    block = (b1_size, b2_size, 1)
    grid = (npairs/b1_size, npairs/b2_size)

    kernel_source = kernels.srt(nsamp)
    mod = SourceModule(kernel_source)
    func = mod.get_function('srtKernel')

    srt_gpu = cuda.mem_alloc( nsamp*npairs*int32 )

    func(exp_gpu, gmap_gpu, srt_gpu )
    
    return srt_gpu
   

def rankTemplate(  srt_gpu, sample_map_gpu, nsamples, neighbors, npairs, b1_size, b2_size):
    """
    srt_gpu is (npairs, nsamples)
    sample_map_gpu is (neighbors, nsamples)
    """

    if debug:
        assert npairs%b1_size == 0
        assert nsamples%b2_size == 0
    
    block = (b1_size, b2_size, 1)
    grid = (npairs/b1_size, nsamples/b2_size)
    
    kernel_source = kernels.rt( neighbors, nsamples )
    mod = SourceModule(kernel_source)
    func = mod.get_function('rtKernel')

    rt_gpu = cuda.mem_alloc( nsamp*npairs*int32 )

    func( srt_gpu, sample_map_gpu, rt_gpu)

    return rt_gpu
    
