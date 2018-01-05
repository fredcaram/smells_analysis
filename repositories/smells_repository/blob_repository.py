import pandas as pd

from repositories.metrics_repository.class_metrics_repository import  class_metrics_repository
from repositories.smells_repository.base_smells_repository import base_smells_repository


class blob_repository(base_smells_repository):
    def __init__(self):
        base_smells_repository.__init__(self)
        self.handled_smell_types = ["Blob"]
        self.metrics_repository = class_metrics_repository()
        self.cache_file_name = "blob"

    def get_cache_file_name(self):
        return self.cache_file_name


    def get_handled_smell_types(self):
        return self.handled_smell_types


    def get_metrics_dataframe(self, prefix, dataset_id, smell):
        return self.metrics_repository.get_metrics_dataframe(prefix, dataset_id)


    def convert_smells_list_to_df(self, smells):
        smells_by_type = [{"instance": smell["instance"].replace(';', '').replace('.java', ''), "smell_type": smell["type"]} for smell in
                          smells]
        smells_df = pd.DataFrame(smells_by_type)
        return smells_df


