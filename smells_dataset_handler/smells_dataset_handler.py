import os, glob
import pandas as pd
from handle_landfill_data.mongodb_helper import mongodb_helper

class smells_dataset_handler:
    def __init__(self):
        self.metrics_dir = "metrics_files/"
        self.projects = ["Apache Ant"]



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
        smells_df = pd.DataFrame(smells)
        combined_df = metrics_df.merge(smells_df, how="left", left_on="Class", right_on="instance")
        return combined_df