import pandas as pd
import re

from repositories.metrics_repository.method_metrics_repository import method_metrics_repository
from repositories.smells_repository.base_smells_repository import base_smells_repository


class method_smells_repository(base_smells_repository):
    def __init__(self):
        base_smells_repository.__init__(self)
        self.handled_smell_types = ["LongMethod", "FeatureEnvy"]
        self.metrics_repository = method_metrics_repository()


    def get_handled_smell_types(self):
        return self.handled_smell_types


    def get_metrics_dataframe(self, prefix):
        return self.metrics_repository.get_metrics_dataframe(prefix)

    def get_only_method_part(self, instance):
        m = re.match("(.*);.*", instance)
        if m is None:
            print("Warning: Couldn't extract method from {0}".format(instance))
            return instance
        method = m.group(1)
        return method

    def convert_smells_list_to_df(self, smells):
        smells_by_type = [{"instance": self.get_only_method_part(smell["instance"]), "smell_type": smell["type"]} for smell in
                          smells]
        smells_df = pd.DataFrame(smells_by_type)
        return smells_df
