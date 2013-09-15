def srt(nsamples):
    base = """
    __global__ void srtKernel( float * d_expression, int *gene_map, int * srt){
        int g_pair = blockIdx.x*blockDim.x + threadIdx.x;//(0,npairs-1)
        int sample = blockIdx.y*blockDim.y + threadIdx.y;//(0,%i-1)
       
        int gm1 = gene_map[ 2*g_pair ];
        int gm2 = gene_map[ 2*g_pair + 1 ];
        
        
        float gene_1_exp = d_expression[gm1*%i + sample];

        float gene_2_exp = d_expression[gm2*%i + sample];
        srt[%i*g_pair + sample] = (int)(gene_1_exp < gene_2_exp);
    }


    """ % tuple([nsamples]*4)
    return base



def rt( neighbors, nsamples ):
    base = """
    __global__ void rtKernel( int * srt,  int * sample_map, int * rt){
        int pair_gid = blockIdx.x*blockDim.x + threadIdx.x;
        int samp_gid = blockIdx.y*blockDim.y + threadIdx.y;
        int sm_off = samp_gid*%i;    
        
        int count = 0;
        #pragma unroll
        for(int i = 0; i<%i; i++){
            count += srt[%i*pair_gid + sample_map[sm_off + i ]];
        }
        rt[ %i*pair_gid + samp_gid]   = %i < count;

    }
    """ % (neighbors, neighbors, nsamples, nsamples, neighbors/2)


    return base

def rms( nsamples, nnets ):
    base = """
    __global__ void rmsKernel( int * rt, int * srt,  int * net_map, float * rms){
        int net =  blockIdx.x*blockDim.x + threadIdx.x;
        int sample = blockIdx.y*blockDim.y + threadIdx.y;
        int counter = 0;
        for( int i=net_map[2*net]; i< net_map[2*net + 1]; i++){
            counter += rt[%i * i + sample] == srt[%i * i + sample]
        }
        if (counter > 0){//counter == 0 if net is in the buffer zone
            rms[net*%i + sample ] = (float)counter/(float)(net_map[2*net] - net_map[2*net]);
        }

    } """ % (nsamples, nsamples, nnets)


if __name__ == "__main__":
    print srt(10)

    print rt(5, 10)
