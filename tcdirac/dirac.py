import static
import debug
import scipy.misc
from pandas import DataFrame
import itertools
import numpy as np
import os
import os.path as op
import logging

def getSRT( df ):
    """
    Given a dataframe(index=gene_names, columns=sample_ids) this method computes the sample rank templates for all samples
    Returns a dataframe(index=gene_names, columns=sample_ids)
    """

    ic = itertools.combinations

    genes = df.index
    srt_rows = scipy.misc.comb(len(genes),2,exact=True)#num rows in new df
    index = ["%s < %s" % (g1,g2) for g1, g2 in  ic(genes,2)]
    #make empty dataframe
    srt = DataFrame(np.zeros((srt_rows,len(df.columns)),dtype=int),
            columns=df.columns, 
            index=index)

    #this should be gpu
    for sample in df.columns:
        c = df[sample]
        idx = 0
        for g1,g2 in ic(genes,2):
            if c[g1] < c[g2]:
                srt[sample][index[idx]] = 1
            idx += 1
    return srt       

def getRT( df ):
    """
    Given a dataframe containing only the samples you want to generate a rank template for
    Returns a Series with rt
    
    """
    nsamps = df.shape[1]

    if debug.DEBUG:
        assert(nsamps == len(df.columns))
    
    return df.sum(1).apply(lambda x: 0 if 2*x < nsamps else 1)   

def getRMS( srts, rt):
    """
    Given dataframe with sample rank templates and a rank template returns the rank matching score
    """
    
    try:
        return srts.apply(lambda x: x==rt, axis=0).applymap(lambda x: 1 if x else 0).sum()/float(rt.shape[0])
    except TypeError:
        #srts is a Series object (a single srt) not a dataframe
        return (srts == rt).apply(lambda x: 1 if x else 0).sum()/float(rt.shape[0])

