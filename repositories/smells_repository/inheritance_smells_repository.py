import pandas as pd
from repositories.metrics_repository.parallel_inheritance_metrics_repository import parallel_inheritance_metrics_repository

from repositories.metrics_repository.class_metrics_repository import class_metrics_repository
from repositories.smells_repository.base_smells_repository import base_smells_repository
from repositories.metrics_repository.metrics_repository_helper import extract_class_from_method
import re


class inheritance_smells_repository(base_smells_repository):
    def __init__(self):
        base_smells_repository.__init__(self)
        self.metrics_dir = "change_history"
        self.handled_smell_types = ["ParallelInheritance"]
        self.file_name = "parallelInheritance"
        self.parallel_inheritance_metrics_repository = parallel_inheritance_metrics_repository()
        self.class_metrics_repo = class_metrics_repository()
        self.class_metrics_repo.metrics_reloaded_class_metrics = ["ck"]
        self.cache_file_name = "parallelInheritance"

    def get_cache_file_name(self):
        return self.cache_file_name

    def get_handled_smell_types(self):
        return self.handled_smell_types

    def get_metrics_dataframe(self, prefix, dataset_id, smell):
        history_metrics = self.parallel_inheritance_metrics_repository.get_metrics_dataframe(prefix)
        class_metrics = self.class_metrics_repo.get_metrics_dataframe(prefix, dataset_id)
        combined_metrics = pd.concat([history_metrics, class_metrics])
        return combined_metrics


    def get_instance(self, instance, pos):
        return self.get_class_part(instance, pos)

    def clean_class(self, method):
        clean_class = method.replace(";", " ").replace(" ", "").replace(".java", "")
        clean_class = re.sub(r'\(.*\).*', "", clean_class)
        clean_class = extract_class_from_method(clean_class)
        return clean_class


    def get_class_part(self, instance, pos):
        regex_match = re.match("^(.+[;])(.+)$", instance)
        if regex_match is None:
            class_ = instance
        else:
            class_ = regex_match.group(pos)

        class_ = self.clean_class(class_)
        return class_


    def convert_smells_list_to_df(self, smells):
        smells_by_type1 = [{"instance": self.get_instance(smell["instance"], 1), "smell_type": smell["type"]} for smell in
                          smells]
        smells_by_type2 = [{"instance": self.get_instance(smell["instance"], 2), "smell_type": smell["type"]} for smell in
                          smells]
        smells_df = pd.DataFrame(smells_by_type1 + smells_by_type2)
        return smells_df