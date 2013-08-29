import time
from mpi4py import MPI
import numpy as np
class MPIProfiler:
    START = 123
    END = 124
    def __init__(self):
        self.log = []
        self.meta = {}
        
    def start(self,label):
        self.log.append((label,MPIProfiler.START, time.time()))

    def end(self,label):
        self.log.append((label,MPIProfiler.END, time.time()))

    def addMeta(self, key, value):
        self.meta[key] = value

    def printLog(self,ranks = None, acc=False):
        comm = MPI.COMM_WORLD
        profilers = comm.gather(self)
        if comm.rank == 0:
            profilers.sort(key=lambda x: x.meta['rank']) 
            for profiler in profilers:
                if ranks is None or profiler.meta['rank'] in ranks:
                    for key, value in profiler.meta.iteritems():
                        print "%s: [%s]" % (key, str(value))
                    print "-"*20
                    for label,rt in profiler.parseLog(acc=acc):
                        print "\t%s\t[%s]" % (label,str(rt))


    def parseLog(self, acc=False):
        parsed = []
        start = {}
        for lab, sore, t in self.log :
            if sore == MPIProfiler.START:
                start[lab] = t
            else:
                parsed.append((lab, t-start[lab]))
        if acc:
            acc_parsed = {}
            for lab,t in parsed:
                if lab not in acc_parsed:
                    acc_parsed[lab] = []
                acc_parsed[lab].append(t)
            new_parsed = []
            for lab,t in parsed:
                if acc_parsed[lab] is not None:
                    total = sum(acc_parsed[lab])
                    m = max(acc_parsed[lab])
                    med = np.median(acc_parsed[lab])
                    num = len(acc_parsed[lab])
                    out = "total[%f]\t count[%i]\tmax[%f]\tmed[%f]" % (total, num, m, med)
                    new_parsed.append((lab, out))
                    acc_parsed[lab] = None
            parsed = new_parsed
        return parsed
            
    

    
