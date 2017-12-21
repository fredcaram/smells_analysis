import pandas as pd
from repositories.metrics_repository.history_change_metrics_repository import history_change_metrics_repository

from repositories.metrics_repository.class_metrics_repository import class_metrics_repository
from repositories.smells_repository.base_smells_repository import base_smells_repository


class relationship_smells_repository(base_smells_repository):
    def __init__(self):
        base_smells_repository.__init__(self)
        self.metrics_dir = "change_history"
        self.handled_smell_types = ['ShotgunSurgery',"DivergentChange"]
        self.file_name = "methodChanges"
        self.history_change_metrics_repo = history_change_metrics_repository()
        self.class_metrics_repo = class_metrics_repository()
        self.class_metrics_repo.metrics_reloaded_class_metrics = ["ck"]
        self.cache_file_name = "relationships"

    def get_cache_file_name(self):
        return self.cache_file_name

    def get_handled_smell_types(self):
        return self.handled_smell_types

    def get_metrics_dataframe(self, prefix, dataset_id):
        history_metrics = self.history_change_metrics_repo.get_metrics_dataframe(prefix)
        class_metrics = self.class_metrics_repo.get_metrics_dataframe(prefix, dataset_id)
        combined_metrics = pd.concat([history_metrics, class_metrics])
        return combined_metrics


    def convert_smells_list_to_df(self, smells):
        smells_by_type = [{"instance": smell["instance"].replace(';', '').replace('.java', ''), "smell_type": smell["type"]} for smell in
                          smells]
        smells_df = pd.DataFrame(smells_by_type)
        return smells_df