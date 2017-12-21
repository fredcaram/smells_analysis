import abc
import os

import pandas as pd

from messages import error_messages
from handle_landfill_data.mongodb_helper import mongodb_helper


class base_smells_repository:
    def __init__(self, metaclass=abc.ABCMeta):
        self.metrics_dir = "metrics_files"
        #self.metrics_reloaded_package_metrics = ["martin"]
        #self.metrics_reloaded_metrod_metrics = ["method_complexity"]


    @abc.abstractproperty
    def get_handled_smell_types(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_handled_smell_types'))

    @abc.abstractproperty
    def get_cache_file_name(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_cache_file_name'))

    def get_trated_cache_file_name(self):
        return "dataset_cache/{0}.csv".format(self.get_cache_file_name())

    @abc.abstractmethod
    def get_metrics_dataframe(self, prefix, dataset_id):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_metrics_dataframe'))


    @abc.abstractmethod
    def convert_smells_list_to_df(self, smells):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('convert_smells_list_to_df'))


    def get_annotated_smells_df(self, db, project_id):
        smells = list(db.smells.find({"project_id": project_id, "type": {"$in": self.get_handled_smell_types()}}))
        if len(smells) == 0:
            return pd.DataFrame()
        smells_df = self.convert_smells_list_to_df(smells)
        smells_df = self.get_smells_dummies(smells_df)
        return smells_df


    def get_smells_dummies(self, smells_df:pd.DataFrame):
        assert "smell_type" in smells_df.columns.values
        smells_dummies = pd.get_dummies(smells_df["smell_type"])
        smells_df = pd.concat([smells_df, smells_dummies], axis=1)
        smells_df = smells_df.drop("smell_type", axis=1)
        return smells_df


    def get_smells_dataset_by_project_id(self, project_id, dataset_id):
        db = mongodb_helper().get_db()
        project = db.smells_projects.find_one({"id": project_id, "dataset_id": dataset_id, "types": {"$elemMatch": {"type": {"$in": self.get_handled_smell_types()}}}})
        if project is None:
            return pd.DataFrame()
        metrics_df = self.get_metrics_dataframe(project["prefix"], dataset_id)
        smells_df = self.get_annotated_smells_df(db, project_id)

        if len(smells_df) == 0 or len(metrics_df) == 0:
            return metrics_df

        combined_df = self.merge_metrics_with_annotation(metrics_df, smells_df)

        return combined_df

    def merge_metrics_with_annotation(self, metrics_df, smells_df):
        assert "instance" in smells_df.columns.values
        smells_grouped_by_class = smells_df.groupby("instance").max().reset_index()
        metrics_df_grouped_by_class = metrics_df.groupby("instance").max().reset_index()
        combined_df = metrics_df_grouped_by_class.merge(smells_grouped_by_class, how="left", left_on="instance", right_on="instance")
        return combined_df

    def get_smells_dataset_from_projects(self, project_ids, dataset_ids):
        projects_df = pd.DataFrame()
        cache_file = self.get_trated_cache_file_name()

        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, header=0)

        for dataset_id in dataset_ids:
            for project_id in project_ids:
                df = self.get_smells_dataset_by_project_id(project_id, dataset_id)
                projects_df = pd.concat((projects_df, df), ignore_index=True)

        projects_df.fillna(0, inplace=True)
        projects_df.to_csv(cache_file)

        return projects_df