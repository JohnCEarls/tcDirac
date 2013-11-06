debug = False 
import kernels
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import numpy as np
def sampleRankTemplate( exp_gpu, gmap_gpu, srt_gpu, nsamp, npairs, pairs_block_size, sample_block_size):
    """
    nsamp  is the columns dim (shape[1]) of exp_gpu
    npairs is the length of gmap_gpu (shape[0])
    """

    if debug:
        assert npairs%pairs_block_size == 0
        assert nsamp%sample_block_size == 0
        print "dirac.debug is on"
    
    block = (pairs_block_size, sample_block_size, 1)
    grid = (npairs/pairs_block_size, nsamp/sample_block_size)

    kernel_source = kernels.srt(nsamp)
    mod = SourceModule(kernel_source)
    func = mod.get_function('srtKernel')


    func(exp_gpu, gmap_gpu, srt_gpu, block=block, grid=grid )

    
   

def rankTemplate(  srt_gpu, sample_map_gpu,rt_gpu, nsamples, neighbors, npairs, pairs_block_size, sample_block_size):
    """
    srt_gpu is (npairs, nsamples)
    sample_map_gpu is (neighbors, nsamples)
    """

    if debug:
        assert npairs%pairs_block_size == 0
        assert nsamples%sample_block_size == 0
    
    block = (pairs_block_size, sample_block_size, 1)
    grid = (npairs/pairs_block_size, nsamples/sample_block_size)
    
    kernel_source = kernels.rt( neighbors, nsamples )
    mod = SourceModule(kernel_source)
    func = mod.get_function('rtKernel')

    func( srt_gpu, sample_map_gpu, rt_gpu, block=block, grid=grid)

def rankMatchingScores( srt_gpu, rt_gpu, rms_gpu, nmap_gpu, nsamples, nnets, sample_block_size, nets_block_size):
    
    block = (nets_block_size, sample_block_size, 1)
    grid = ( nnets/nets_block_size, nsamples/sample_block_size)

    kernel_source = kernels.rms( nsamples, nnets )
    mod = SourceModule( kernel_source )
    func = mod.get_function('rmsKernel')
    
    func( rt_gpu, srt_gpu, nmap_gpu, rms_gpu, block=block, grid=grid )

