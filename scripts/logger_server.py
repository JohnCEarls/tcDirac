import inspect, os, os.path,sys
if os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) == '/home/sgeadmin/hdproject/tcDirac/scripts':
    #if we are running this from dev dir, need to add tcdirac to the path
    #should really be removed when module is installed and not in dev
    sys.path.append('/home/sgeadmin/hdproject/tcDirac')
import tcdirac.debug
with open('logger.pid', 'w') as l:
    l.write(str(os.getpid()))
tcdirac.debug.startLogger()
