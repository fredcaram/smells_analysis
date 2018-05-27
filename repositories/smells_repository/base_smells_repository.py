import abc
import os

import pandas as pd

from handle_landfill_data.mongodb_helper import mongodb_helper
from messages import error_messages


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

    @abc.abstractmethod
    def get_metrics_dataframe(self, prefix, dataset_id, smell):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_metrics_dataframe'))


    @abc.abstractmethod
    def convert_smells_list_to_df(self, smells):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('convert_smells_list_to_df'))

    def get_trated_cache_file_name(self, smell):
        return "dataset_cache/{0}_{1}.csv".format(self.get_cache_file_name(), smell)

    def merge_metrics_with_annotation(self, metrics_df, smells_df):
        assert "instance" in smells_df.columns.values
        #smells_grouped_by_class = smells_df.groupby("instance").max().reset_index()
        metrics_df_grouped_by_class = metrics_df.groupby("instance").max().reset_index()
        combined_df = metrics_df_grouped_by_class.merge(smells_df, how="left", left_on="instance", right_on="instance")
        return combined_df

    def get_annotated_smells_df(self, project_id, smell):
        file_name = "smells_cache/smell_oracle_{0}_{1}.csv".format(smell, project_id)

        if not os.path.exists(file_name):
            db = mongodb_helper().get_db()
            smells = list(db.smells.find({"project_id": project_id, "type": smell}))
            if len(smells) == 0:
                smells_df = pd.DataFrame()
                smells_df.to_csv(file_name)
                return smells_df

            smells_df = self.convert_smells_list_to_df(smells)
            smells_df.to_csv(file_name)
        else:
            smells_df = pd.read_csv(file_name)
            if len(smells_df) == 0:
                return smells_df

        smells_df = self.get_smells_dummies(smells_df)
        return smells_df


    def get_smells_dummies(self, smells_df:pd.DataFrame):
        assert "smell_type" in smells_df.columns.values
        smells_dummies = pd.get_dummies(smells_df["smell_type"])
        smells_df = pd.concat([smells_df, smells_dummies], axis=1)
        smells_df = smells_df.drop("smell_type", axis=1)
        return smells_df


    def get_smells_dataset_by_project_id(self, smell, project_id, dataset_id):
        file_name = "smells_cache/smell_project_{0}_{1}_{2}.csv".format(smell, project_id, dataset_id)

        if not os.path.exists(file_name):
            db = mongodb_helper().get_db()
            project = db.smells_projects.find_one({"id": project_id, "dataset_id": dataset_id, "types.type": smell})

            if project is None:
                project_df = pd.DataFrame()
            else:
                project["types"] = smell
                project_df = pd.DataFrame(project, index=[0])
            project_df.to_csv(file_name)
        else:
            project_df = pd.read_csv(file_name)

        if len(project_df) == 0:
            return project_df

        smells_df = self.get_annotated_smells_df(project_id, smell)
        metrics_df = self.get_metrics_dataframe(project_df["prefix"][0], dataset_id, smell)

        if len(smells_df) == 0 or len(metrics_df) == 0:
            return metrics_df

        combined_df = self.merge_metrics_with_annotation(metrics_df, smells_df)

        return combined_df

    def get_smells_dataset_from_projects(self, smell, project_ids, dataset_ids):
        projects_df = pd.DataFrame()
        cache_file = self.get_trated_cache_file_name(smell)

        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, header=0)

        for dataset_id in dataset_ids:
            for project_id in project_ids:
                df = self.get_smells_dataset_by_project_id(smell, project_id, dataset_id)
                df["project_id"] = project_id
                projects_df = pd.concat((projects_df, df), ignore_index=True)

        projects_df.fillna(0, inplace=True)
        projects_df.to_csv(cache_file)


        return projects_df