import dirac
import data
import pycuda as cuda
import numpy as np
def runDirac( expression_matrix, gene_map, sample_map, network_map, sample_block_size, npairs_block_size, nets_block_size ):
    exp = data.Expression( expression_matrix )
    exp.toGPU( sample_block_size )

    gm = data.GeneMap( gene_map )
    gm.toGPU( npairs_block_size )
    
    srt = data.SampleRankTemplate( exp.orig_nsamples, gm.orig_npairs )
    srt.toGPU( sample_block_size, npairs_block_size )

    sm = data.SampleMap( sample_map )
    sm.toGPU( sample_block_size )
    
    rt = data.RankTemplate( exp.orig_nsamples, gm.orig_npairs )
    rt.toGPU( sample_block_size, npairs_block_size )

    nm = data.NetworkMap( network_map )
    nm.toGPU( nets_block_size )

    rms = data.RankMatchingScores( nm.orig_nnets, exp.orig_nsamples)
    rms.toGPU( sample_block_size, nets_block_size )

    dirac.sampleRankTemplate( exp.gpu_data, gm.gpu_data, srt.gpu_data, exp.buffer_nsamples, gm.buffer_npairs, npairs_block_size, sample_block_size)
    
    dirac.rankTemplate( srt.gpu_data, sm.gpu_data, rt.gpu_data, srt.buffer_nsamples, sm.orig_kneighbors, gm.buffer_npairs, npairs_block_size, sample_block_size)

    dirac.rankMatchingScores( srt.gpu_data, rt.gpu_data, rms.gpu_data, nm.gpu_data, srt.buffer_nsamples, nm.buffer_nnets, sample_block_size, nets_block_size)

    return (srt, rt, rms)

if __name__ == "__main__":
    import numpy as np
    import pandas
    import itertools
    import random
    import pycuda.driver as cuda
    import time
    import scipy.misc

    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()


    n = 500
    gn = 10000
    neighbors =  10

    samples = map(lambda x:'s%i'%x, range(n))
    genes = map(lambda x:'g%i'%x, range(gn))
    g_d = dict([(gene,i) for i,gene in enumerate(genes)])
    gm_text = []   
    gm_idx = []


    exp = np.random.rand(len(genes),len(samples))
    exp_df = pandas.DataFrame(exp,dtype=float, index=genes, columns=samples)

    net_map = [0]
   
    for i in range(250):
        n_size = random.randint(5, 50)
        net_map.append(net_map[-1] + scipy.misc.comb(n_size,2, exact=1))
        net = random.sample(genes,n_size)
        for g1,g2 in itertools.combinations(net,2):
            gm_text.append("%s < %s" % (g1,g2))
            gm_idx += [g_d[g1],g_d[g2]]

    #data
    expression_matrix = exp
    gene_map = np.array(gm_idx)
    #print gene_map[:20]
    sample_map = np.random.randint(low=0,high=len(samples), size=(neighbors, len(samples)))
    #print sample_map[:,:3]
    network_map = np.array(net_map)
    #print network_map[:10]
    #settings
    sample_block_size = 32 
    npairs_block_size = 16
    nets_block_size = 4 
    
 
    st = time.time()
    srt,rt,rms = runDirac( expression_matrix, gene_map, sample_map, network_map, sample_block_size, npairs_block_size, nets_block_size )
    print "running time", time.time() - st
    srt.fromGPU()
    print "pairs:", srt.res_npairs
    rt.fromGPU()
    rms.fromGPU()
    print rms.res_data[:10,:10]
    ctx.pop()
