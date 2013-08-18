import scipy.misc
from pandas import DataFrame
import itertools
import numpy as np
def getSRT( df ):
    genes = df.index
    srt_rows = scipy.misc.comb(len(genes),2,exact=True)
    srt = DataFrame(np.empty((srt_rows,df.shape[0]),dtype=int) columns=df.columns, index = ["%s < %s" % (g1,g2) for g1, g2 in  itertools.combinations(genes,2)])
    first = False

    for sample in df.columns:
        c = df[sample]
        for g1,g2 in itertools.combinations(genes,2):
            srt[sample]["%s < %s" % (g1,g2)] = 1 if c[g1] < c[g2]  else 0
    return srt       
def getRT( df ):
    nsamps = df.shape[0]
    return df.sum(1).apply(lambda x: 0 if 2*x < nsamps else 1)   

def getRMS( srts, rt):
    return srts.apply(lambda x: x==rt, axis=0).applymap(lambda x: 1 if x else 0).sum()/float(rt.shape[0])
