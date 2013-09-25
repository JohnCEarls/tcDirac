import dirac
import data
import pycuda as cuda
import numpy as np
def runDirac( expression_matrix, gene_map, sample_map, network_map, sample_block_size, npairs_block_size, nets_block_size,rms_only=False ):
    exp = data.Expression( expression_matrix )

    gm = data.GeneMap( gene_map )
    
    srt = data.SampleRankTemplate( exp.orig_nsamples, gm.orig_npairs )

    sm = data.SampleMap( sample_map )
    
    rt = data.RankTemplate( exp.orig_nsamples, gm.orig_npairs )

    nm = data.NetworkMap( network_map )

    rms = data.RankMatchingScores( nm.orig_nnets, exp.orig_nsamples)

    req_mem = reqMemory(exp, rms,np,rt,sm,srt,gm,nm, sample_block_size, nets_block_size, npairs_block_size )
    
    print "Req. Mem[%f], Avail. Mem[%f]" % (float(req_mem)/(2**30), float(cuda.mem_get_info()[0])/2**30)

    if req_mem > cuda.mem_get_info()[0]*.9 :
        return splitDirac( expression_matrix, gene_map, sample_map, network_map, sample_block_size, npairs_block_size, nets_block_size, rms_only )

    try:
        exp.toGPU( sample_block_size )
        rms.toGPU( sample_block_size, nets_block_size )
        nm.toGPU( nets_block_size )
        rt.toGPU( sample_block_size, npairs_block_size )
        sm.toGPU( sample_block_size )
        srt.toGPU( sample_block_size, npairs_block_size )
        gm.toGPU( npairs_block_size )
    except:
        print "Not enough Memory and missed the calc"
        raise

    dirac.sampleRankTemplate( exp.gpu_data, gm.gpu_data, srt.gpu_data, exp.buffer_nsamples, gm.buffer_npairs, npairs_block_size, sample_block_size)
    
    dirac.rankTemplate( srt.gpu_data, sm.gpu_data, rt.gpu_data, srt.buffer_nsamples, sm.orig_kneighbors, gm.buffer_npairs, npairs_block_size, sample_block_size)

    dirac.rankMatchingScores( srt.gpu_data, rt.gpu_data, rms.gpu_data, nm.gpu_data, srt.buffer_nsamples, nm.buffer_nnets, sample_block_size, nets_block_size)

    return (srt, rt, rms)

def reqMemory(exp, rms,np,rt,sm,srt,gm,nm,sample_block_size, nets_block_size, npairs_block_size ):
    pred = exp.gpu_mem( sample_block_size )
    pred += rms.gpu_mem( sample_block_size, nets_block_size )
    pred += nm.gpu_mem( nets_block_size )
    pred += rt.gpu_mem( sample_block_size, npairs_block_size )
    pred += sm.gpu_mem( sample_block_size )
    pred += srt.gpu_mem( sample_block_size, npairs_block_size )
    pred += gm.gpu_mem( npairs_block_size )
    return pred

def splitDirac( expression_matrix, gene_map, sample_map, network_map, sample_block_size, npairs_block_size, nets_block_size, rms_only=False ):
    gm =  data.GeneMap( gene_map )
    nm = data.NetworkMap( network_map )
    part_point,nm1,nm2 = nm.split()
    gm1,gm2 = gm.split(part_point)
    srt1, rt1, rms1 = runDirac( expression_matrix, gm1.orig_data, sample_map, nm1.orig_data, sample_block_size, npairs_block_size, nets_block_size, rms_only )
    if rms_only:
        srt1 = None
        rt1 = None
        rms1.fromGPU()
        rms1.buffer_data = None
        if rms1.gpu_data is not None:
            rms1.gpu_data.free()
        rms1.gpu_data = None
    else:
        for d in [srt1,rt1,rms1]:
            d.fromGPU()
            if d.gpu_data is not None:
                d.gpu_data.free()
            d.buffer_data = None
            d.gpu_data = None

    srt2,rt2,rms2 = runDirac( expression_matrix, gm2.orig_data, sample_map, nm2.orig_data, sample_block_size, npairs_block_size, nets_block_size,rms_only )

    if rms_only:
        srt2 = None
        rt2 = None
        rms2.fromGPU()
        rms2.buffer_data = None
        if rms2.gpu_data is not None:#happens if rms is from previous split
            rms2.gpu_data.free()
        rms2.gpu_data = None
    else:
        for d in [srt2,rt2,rms2]:
            d.fromGPU()
            if d.gpu_data is not None:
                d.gpu_data.free()
            d.buffer_data = None
            d.gpu_data = None


    if rms_only:
        srt = None
        rt = None
    else:
        srt = data.SampleRankTemplate( srt1.res_nsamples, srt1.res_npairs + srt2.res_npairs)
        srt.res_data = np.zeros( (srt.res_npairs,srt.res_nsamples),dtype=srt1.res_data.dtype)
        srt.res_data[:srt1.res_npairs, :] = srt1.res_data[:,:]
        srt.res_data[srt1.res_npairs:, :] = srt2.res_data[:,:]

        
        rt = data.RankTemplate( rt1.res_nsamples, rt1.res_npairs + rt2.res_npairs)
        rt.res_data = np.zeros( (rt.res_npairs,rt.res_nsamples),dtype=rt1.res_data.dtype)
        rt.res_data[:rt1.res_npairs, :] = rt1.res_data[:,:]
        rt.res_data[rt1.res_npairs:, :] = rt2.res_data[:,:]

    rms = data.RankMatchingScores(rms1.res_nnets + rms2.res_nnets, rms1.res_nsamples)
    rms.res_data = np.zeros((rms.res_nnets, rms.res_nsamples), dtype=rms1.res_data.dtype)
    rms.res_data[:rms1.res_nnets, :] = rms1.res_data[:,:]
    rms.res_data[rms1.res_nnets:, :] = rms2.res_data[:,:]

    return (srt, rt, rms)
    


def testDirac(expression_matrix, gene_map, sample_map, network_map):
    srt = np.zeros((gene_map.shape[0]/2, expression_matrix.shape[1]))
    for i in range(expression_matrix.shape[1]):
        for j in range(gene_map.shape[0]/2):
            g1 = gene_map[2*j]
            g2 = gene_map[2*j +  1]
            if expression_matrix[g1,i] < expression_matrix[g2,i]:
                srt[j,i] = 1
            else:
                srt[j,i] = 0
    rt = np.zeros_like(srt)
    
    for i in range(expression_matrix.shape[1]):
        neigh = sample_map[i,:]
        t = srt[:,neigh].sum(axis=1)
        for j in range(len(t)):
            rt[j,i] = int(len(neigh)/2 < t[j])

    rms_matrix =  np.zeros_like(srt)
    for i in range(expression_matrix.shape[1]):
        for j in range(gene_map.shape[0]/2):
            rms_matrix[j,i] = int(rt[j,i] == srt[j,i])
    rms_final = np.zeros((len(network_map) - 1 , expression_matrix.shape[1]))
    
    for i in range(len(network_map) - 1):
        nstart = network_map[i]
        nend = network_map[i+1]
        rms_final[i,:] = rms_matrix[nstart:nend, :].sum(axis=0)/float(nend-nstart)
        
    return srt, rt, rms_final

                
         

            

if __name__ == "__main__":
    import numpy as np
    import pandas
    import itertools
    import random
    import pycuda.driver as cuda
    import time
    import scipy.misc
    test = False 
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()
    for _ in range(100):

        n = random.randint(500,2000)
        gn = 10000
        neighbors = random.randint(5, 20) 
        nnets = random.randint(50,300)

        samples = map(lambda x:'s%i'%x, range(n))
        genes = map(lambda x:'g%i'%x, range(gn))
        g_d = dict([(gene,i) for i,gene in enumerate(genes)])
        gm_text = []   
        gm_idx = []


        exp = np.random.rand(len(genes),len(samples)).astype(np.float32)
        exp_df = pandas.DataFrame(exp,dtype=float, index=genes, columns=samples)

        net_map = [0]
       
        for i in range(nnets):
            n_size = random.randint(5,300)

            net_map.append(net_map[-1] + scipy.misc.comb(n_size,2, exact=1))
            net = random.sample(genes,n_size)
            for g1,g2 in itertools.combinations(net,2):
                gm_text.append("%s < %s" % (g1,g2))
                gm_idx += [g_d[g1],g_d[g2]]

        #data
        expression_matrix = exp
        gene_map = np.array(gm_idx)
        #print gene_map[:20]
        sample_map = np.random.randint(low=0,high=len(samples), size=(len(samples),neighbors))
        #print sample_map[:,:3]
        network_map = np.array(net_map)
        #print network_map[:10]
        #settings
        sample_block_size = 32 
        npairs_block_size = 16
        nets_block_size = 4 

        print "nsamples", n
        print "kneighbors", neighbors
        print "nnets", nnets
        print "sample_block_size", sample_block_size
        print "npair_block_size", npairs_block_size 
        print "nets_block_size", nets_block_size
        print "npairs", len(gene_map)/2
        num_rep = 1 
        acc = 0.0
        for i in range(num_rep):
            st = time.time()
            srt,rt,rms = runDirac( expression_matrix, gene_map, sample_map, network_map, sample_block_size, npairs_block_size, nets_block_size, True )
            acc += time.time() - st

        print "running time(avg):", acc/num_rep
        if acc > 0:
            print "++ran++"
        if test:
            st = time.time()
            test_srt, test_rt, test_rms = testDirac(expression_matrix, gene_map, sample_map, network_map)
            print "Test running time", time.time() - st
            ap = True
            """
            print "SRT check:", 
            if np.allclose(srt.res_data, test_srt, atol=1e-02):
                print "PASSED"
            else:
                print  "FAILED"
                ap = False
            print "RT  check:",
            if  np.allclose(rt.res_data, test_rt, atol=1e-02):
                print "PASSED"
            else: 
                print  "FAILED"
                ap = False
            """
            print "RMS check:",
            if  np.allclose(rms.res_data, test_rms,  atol=1e-02):
                print "PASSED"
            else: 
                print  "FAILED"
                ap = False
            
            if not ap:
                print "saving tables"
                od = [exp,gene_map, srt.res_data, test_srt, rt.res_data, test_rt,rms.res_data, test_rms,network_map,sample_map]
                ofn = 'exp,gene_map,srt.res_data,test_srt,rt.res_data,test_rt,rms.res_data,test_rms,net_map,samp_map'.split(',')
                ofn = [str(srt.res_npairs) + '-'+ x for x in ofn]
                for fn, d in zip(ofn, od):
                    np.save('/scratch/sgeadmin/' + fn, d)
                print "*"*20
                break
            test_srt,test_rt, test_rms = (None,None,None)

       
    
    ctx.pop()
