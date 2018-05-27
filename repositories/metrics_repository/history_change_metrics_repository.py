import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import OnehotTransactions

from repositories.metrics_repository.base_metrics_repository import base_metrics_repository
from repositories.metrics_repository.metrics_repository_helper import extract_class_from_method, \
    extract_method_without_parameters


class history_change_metrics_repository(base_metrics_repository):
    def __init__(self):
        base_metrics_repository.__init__(self)
        self.metrics_dir = "change_history"
        self.handled_smell_types = ['ShotgunSurgery', "DivergentChange"]
        self.file_name = "methodChanges"
        self.save_association_rules = False
        self.support_by_project = {"apache_james": 0.01,
                                   "apache_tomcat": 0.003,
                                   "cassandra": 0.004,
                                   "default": 0.008}


    def get_metrics_dataframe(self, prefix):
        file = "{0}/{1}/{2}.csv".format(self.metrics_dir, prefix, self.file_name)

        file_df = pd.read_csv(file, sep=";")

        if prefix.startswith("android"):
            metrics_df = self.handle_android_method_change_file(file_df)
        else:
            metrics_df = self.handle_default_method_change_file(file_df)

        metrics_df.columns = ["commit", "instance"]
        #metrics_df["instance"] = list([extract_method_without_parameters(method) for method in metrics_df["instance"].values])
        metrics_df["instance"] = list([extract_class_from_method(method) for method in metrics_df["instance"].values])
        metrics_df = metrics_df.drop_duplicates()

        a_rules_df = self.get_association_rules(metrics_df, prefix)
        if self.save_association_rules:
            a_rules_df.to_csv("logs/assoc_{0}.csv".format(prefix))
        a_rules_df = a_rules_df.drop(["antecedants", "commit"], axis=1, errors="ignore")

        return a_rules_df

    def remove_one_change_only_commit(self, df):
        combined_df = df.groupby("commit")
        cleaned_df = combined_df.filter(lambda c: c["instance"].count() > 1)
        return cleaned_df


    def get_association_rules(self, df, prefix):

        oht = OnehotTransactions()
        #treated_df = self.remove_one_change_only_commit(df)

        data = [list(v["instance"].values) for k, v in df.groupby("commit")]
        oht_data = oht.fit_transform(data)
        oht_df = pd.DataFrame(oht_data, columns=oht.columns_)
        support = self.support_by_project.get(prefix, self.support_by_project["default"])

        print("Generating Apriori for {0} with support {1}".format(prefix, support))
        frequent_itemsets = apriori(oht_df, min_support=support, use_colnames=True)
        #frequent_itemsets = apriori(oht_df, min_support=0.002, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        if len(rules) == 0:
            return df

        one_ante_rule = rules[[len(ante) == 1 for ante in rules["antecedants"]]]
        del(rules)

        one_ante_rule.loc[:, "antecedants"] = one_ante_rule["antecedants"].apply(lambda x: next(iter(x)))

        category = []
        consequent_not_in_association = []
        for i, rule in one_ante_rule.iterrows():
            n_ocurrences = 0
            category.append(len(rule["consequents"]))

            for conseq in rule["consequents"]:
                rules_with_conseq = one_ante_rule[one_ante_rule["antecedants"] == conseq]

                for conseq_from_conseq in list(rules_with_conseq["consequents"]):
                    if not (conseq_from_conseq in list(rule["consequents"]) or
                                    conseq_from_conseq == rule["antecedants"]):
                        n_ocurrences += 1
                        break

            consequent_not_in_association.append(int(n_ocurrences == 0))


        one_ante_rule.loc[:, "cardinality"] = category
        one_ante_rule.loc[:, "consequent_not_in_association"] = consequent_not_in_association

        one_ante_rule = one_ante_rule.drop("consequents", axis=1)
        #one_ante_rule.loc[:, "antecedants"] = list([extract_class_from_method(method) for method in one_ante_rule["antecedants"].values])
        max_ante_rule = one_ante_rule.groupby("antecedants").max().reset_index()

        df = df.merge(max_ante_rule, left_on="instance", right_on="antecedants")

        return df

    def handle_android_method_change_file(self, file_df):
        metrics_df = file_df[file_df["Entity"].values == "METHOD"]
        metrics_df = metrics_df.drop(["Date", "BugFix", "Entity", "Public", "ChangeType"], axis=1, errors="ignore")
        return metrics_df

    def handle_default_method_change_file(self, file_df):
        metrics_df = file_df
        metrics_df = metrics_df.drop(["Entity", "Change"], axis=1, errors="ignore")
        return metrics_df