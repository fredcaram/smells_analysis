import os, glob
import pandas as pd
from handle_landfill_data.mongodb_helper import mongodb_helper

class smells_dataset_handler:
    def __init__(self):
        self.metrics_dir = "metrics_files/"
        self.projects = ["Apache Ant"]
        self.columns_order = ['Class', 'CBO', 'DIT', 'LCOM', 'NOC', 'RFC',
                              'WMC', 'Cyclic', 'Dcy', 'Dcy*', 'Dpt', 'Dpt*',
                              'Blob', 'EagerTest', 'FeatureEnvy', 'GeneralFixture', 'LongMethod']


    def get_metric_files(self, prefix):
        metrics_df = pd.DataFrame()
        current_dir = os.getcwd()
        print(os.getcwd())
        os.chdir(self.metrics_dir)
        print(os.getcwd())

        for file in glob.glob(prefix + "_*"):
            df = pd.read_csv(file, skiprows=1)

            if len(df) == 0:
                continue
            if len(metrics_df) > 0:
                metrics_df = metrics_df.merge(df, left_on="Class", right_on="Class")
            else:
                metrics_df = df

        os.chdir(current_dir)

        for col in metrics_df.iloc[:, 1:].columns.values:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

        return metrics_df

    def rename_column(self, df, col_index, new_name):
        df_columns = df.columns.values
        df_columns[col_index] = new_name
        return df_columns

    def get_smells_dataset_by_project_id(self, project_id):
        db = mongodb_helper().get_db()
        project = db.smells_projects.find_one({"id": project_id})
        metrics_df = self.get_metric_files(project["prefix"])
        smells = list(db.smells.find({"project_id": project_id}))
        smells_by_type = [{"Class":smell["instance"], "smell_type":smell["type"]} for smell in smells]
        smells_df = pd.DataFrame(smells_by_type)
        smells_dummies = pd.get_dummies(smells_df["smell_type"])
        smells_df = pd.concat([smells_df, smells_dummies], axis=1)
        smells_df = smells_df.drop("smell_type", axis=1)
        smells_grouped_by_class = smells_df.groupby("Class").max().reset_index()
        combined_df = metrics_df.merge(smells_grouped_by_class, how="left", left_on="Class", right_on="Class")
        combined_df.fillna(0, inplace=True)
        return combined_df

    def get_smells_dataset_from_projects(self, project_ids):
        projects_df = pd.DataFrame()
        for project_id in project_ids:
            df = self.get_smells_dataset_by_project_id(project_id)
            projects_df = pd.concat((projects_df, df), ignore_index=True)

        projects_df = projects_df.reindex_axis(self.columns_order, axis=1)
        return projects_df