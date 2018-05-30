import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from repositories.metrics_repository.base_metrics_repository import base_metrics_repository
from repositories.metrics_repository.metrics_repository_helper import extract_class_from_path
import os


class parallel_inheritance_metrics_repository(base_metrics_repository):
    def __init__(self):
        base_metrics_repository.__init__(self)
        self.metrics_dir = "change_history"
        self.handled_smell_types = ["ParallelInheritance"]
        self.file_name = "parallelInheritance"
        self.save_association_rules = False
        self.support_by_project = {"apache_io": 0.01,
                                   "apache_logging": 0.01,
                                   "apache_tomcat": 0.003,
                                   "cassandra": 0.013,
                                    "jedit": 0.012,
                                   "eclipse_core": 0.005,
                                    "ant": 0.001,
                                   "default": 0.008}


    def get_metrics_dataframe(self, prefix, dataset_id=None):
        file = "{0}/{1}/{2}.csv".format(self.metrics_dir, prefix, self.file_name)
        if not os.path.isfile(file):
            return pd.DataFrame()

        file_df = pd.read_csv(file, sep=";")

        metrics_df = file_df.drop(["Entity", "Change"], axis=1, errors="ignore")

        metrics_df.columns = ["commit", "instance", "superclass"]
        #metrics_df["instance"] = list([extract_class_from_path(class_) for class_ in metrics_df["instance"].values])

        metrics_df.loc[:,"commit_superclass"] = \
            np.char.add(list(metrics_df["commit"].astype(str).values),
                        list(metrics_df["superclass"].values))

        metrics_df = metrics_df.drop(["commit", "superclass"], axis=1, errors="ignore")

        metrics_df = metrics_df.drop_duplicates()

        a_rules_df = self.get_association_rules(metrics_df, prefix)
        if self.save_association_rules:
            a_rules_df.to_csv("logs/assoc_{0}.csv".format(prefix))
        a_rules_df = a_rules_df.drop(["antecedants", "commit_superclass"], axis=1, errors="ignore")

        return a_rules_df

    def remove_one_change_only_commit(self, data):
        return [d for d in data if len(d) > 0]


    def get_association_rules(self, df, prefix):

        oht = TransactionEncoder()

        data = []
        superclasses = []

        for k, v in df.groupby("commit_superclass"):
            data.append(list(v["instance"].values.astype(str)))
            #superclasses.append(list(v["superclass"].values))

        #data = self.remove_one_change_only_commit(data)
        #superclasses = np.ravel(superclasses)
        #superclasses = np.unique(superclasses)

        oht_data = oht.fit_transform(data)
        oht_df = pd.DataFrame(oht_data, columns=oht.columns_)
        support = self.support_by_project.get(prefix, self.support_by_project["default"])

        print("Generating Apriori for {0} with support {1}".format(prefix, support))
        frequent_itemsets = apriori(oht_df, min_support=support, use_colnames=True)
        #frequent_itemsets = apriori(oht_df, min_support=0.002, use_colnames=True)
        #if len(frequent_itemsets) == 0:
            #return df

        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        if len(rules) == 0:
            return df

        one_ante_rule = rules[[len(ante) == 1 for ante in rules["antecedants"]]]
        #one_ante_rule = rules[[len(ante) == 1 and ante in superclasses for ante in rules["antecedants"]]]
        one_ante_rule.loc[:,"antecedants"] = one_ante_rule["antecedants"].apply(lambda x: next(iter(x)))
        one_ante_rule = one_ante_rule.drop("consequents", axis=1)
        max_ante_rule = one_ante_rule.groupby("antecedants").max().reset_index()

        df = df.merge(max_ante_rule, left_on="instance", right_on="antecedants")

        return df