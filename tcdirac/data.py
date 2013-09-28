from boto.dynamodb2.table import Table 
from pandas import DataFrame
import pandas
import logging
import os.path

class SourceData:
    """
    A repository for data
    """
    def __init__(self):
        logging.info("creating SourceData object")
        self.source_dataframe = None
        self.net_info = None
        self.genes = set()
    

    def load_dataframe( self, data_frame_source='/scratch/sgeadmin/hddata/trimmed_dataframe.pandas' ):
        logging.info("Loading existing expression dataframe[%s]" % data_frame_source)
        try:
            self.source_dataframe = pandas.read_pickle(data_frame_source)
            self.genes = set(self.source_dataframe.index)
        except Exception as e:
            logging.error("Error loading [%s]" % data_frame_source)
            logging.error(str(e))
            raise(e)
    
    def load_net_info(self,  table_name="net_info_table",source_id="c2.cp.biocarta.v4.0.symbols.gmt"):
        self.net_info = NetworkInfo(table_name,source_id)
        self.initGenes()


    def getExpression(self, sample_ids):
        df = self.source_dataframe
        return df.loc[:,sample_ids]

    def getPathways(self):
        return self.net_info.getPathways()

    def getGenes(self, pathway_id):
        ni = self.net_info
        genes = ni.getGenes(pathway_id)
        if not ni.isClean(pathway_id):
            gset =  self.genes
            genes = [g for g in genes if g in gset]
            ni.updateGenes( pathway_id, genes )
        return self.net_info.getGenes(pathway_id)

    def initGenes(self):
        for pw in self.getPathways():
            self.getGenes(pw)
            
class MetaInfo:
    def __init__(self, meta_file):
        self.metadata = pandas.io.parsers.read_table(meta_file)
        self.metadata.index = self.metadata['sample_id']
        #for i in self.metadata.index:
        #    assert( self.metadata['sample_id'][i] == i)

    def getSampleIDs(self, strain, allele=None):
        md = self.metadata
        if allele is None:
            return md[md['strain'] == strain]['sample_id'].tolist()
        else:
            return md[(md['strain'] == strain) & (md['allele_nominal'] == allele)]['sample_id'].tolist()

    def getStrains(self):
        return self.metadata['strain'].unique().tolist()

    def getNominalAlleles(self, strain=None):
        md = self.metadata
        if strain is None:
            return md['allele_nominal'].unique().tolist()
        else:
            return md[md['strain'] == strain]['allele_nominal'].unique().tolist()

    def getAge(self, sample_id):
        return self.metadata['age'][sample_id]


class NetworkInfo:
    """
    Stores network information
    """
    def __init__(self, table_name="net_info_table",source_id="c2.cp.biocarta.v4.0.symbols.gmt"):
        self.table = Table(table_name)
        self.source_id = source_id
        self.gene_map = {}
        #clean refers to the genes being filtered to match the genes
        #available in the expression file
        self.gene_clean = {}
        self.pathways = []

    def getGenes(self, pathway_id, cache=True):
        if pathway_id not in self.gene_map:
            table = self.table
            source_id = self.source_id
            logging.info("Getting network info [%s.%s.%s]" % (table.table_name, source_id, pathway_id))
            nit_item = table.get_item(src_id=source_id, pw_id=pathway_id)
            self.gene_map[pathway_id] = nit_item['gene_ids'][6:].split('~:~')
            self.gene_clean[pathway_id] = False
        return self.gene_map[pathway_id]

    def getPathways(self):
        if len(self.pathways) == 0:
            pw_ids = self.table.query(src_id__eq=self.source_id, attributes=('pw_id','gene_ids'))
            #simple load balancing
            t = [(len(pw['gene_ids'].split('~:~')), pw['pw_id']) for pw in pw_ids]
            t.sort()
            self.pathways = [pw for _,pw in t]            
            for pw in pw_ids:
                self.gene_map[pw['pw_id']] =pw['gene_ids'][6:].split('~:~')
                self.gene_clean[pathway_id] = False
                
             
        return self.pathways
        

    def isClean(self, pathway_id):
        return self.gene_clean[pathway_id]

    def updateGenes(self, pathway_id, genes):
        """
        Updates gene list for pathway_id to genes(list)
        """
        self.gene_map[pathway_id] = genes
        self.gene_clean[pathway_id] = True

    def clearCache(self):
        self.gene_map = {}
        self.gene_clean = {}


if __name__ == "__main__":
    #getNetworkExpression( "c2.cp.biocarta.v4.0.symbols.gmt", "BIOCARTA_AKAPCENTROSOME_PATHWAY")
    local_data_dir = '/scratch/sgeadmin/hddata/'
    meta_file = os.path.join(local_data_dir, 'metadata.txt')
    """
    ni = NetworkInfo()
    pathway_id = 'BIOCARTA_AKAPCENTROSOME_PATHWAY'
    print ni.getGenes('BIOCARTA_AKAPCENTROSOME_PATHWAY')
    sd = SourceData()
    sd.load_dataframe()
    sd.load_net_info()    
    p_exp = sd.getExpression(pathway_id)
    print p_exp"""
    mi = MetaInfo(meta_file)
    print mi.metadata
    for sid in mi.getSampleIDs('FVB'):
        print mi.getAge(sid)

