import time
from mpi4py import MPI
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

    def printLog(self,ranks = None):
        comm = MPI.COMM_WORLD
        profilers = comm.gather(self)
        if comm.rank == 0:
            profilers.sort(key=lambda x: x.meta['rank']) 
            for profiler in profilers:
                if ranks is None or profiler.meta['rank'] in ranks:
                    for key, value in profiler.meta.iteritems():
                        print "%s: [%s]" % (key, str(value))
                    print "-"*20
                    for label,rt in profiler.parseLog():
                        print "\t%s : [%f]" % (label,rt)


    def parseLog(self):
        parsed = []
        start = {}
        for lab, sore, t in self.log :
            if sore == MPIProfiler.START:
                start[lab] = t
            else:
                parsed.append((lab, t-start[lab]))
        return parsed
            
    

    
