from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np

from models.model_base import model_base
from smells_repository.relationships_smells_repository import \
    relationship_smells_repository


class history_based_model(model_base):
    def __init__(self, classifier=DecisionTreeClassifier()):
        self.classifier = classifier
        model_base.__init__(self)
        self.history_based_smells = ['ShotgunSurgery', "DivergentChange"]#, "ParallelInheritance"

    def get_classifier(self):
        return self.classifier

    def get_dataset(self):
        df = relationship_smells_repository().get_smells_dataset_from_projects(self.projects_ids)
        a_rules_df = self.get_association_rules(df)
        a_rules_df = a_rules_df.drop(["antecedants", "commit"], axis=1)
        return a_rules_df

    def get_association_rules(self, df):
        oht = OnehotTransactions()

        data = [list(v["instance"].values) for k, v in df.groupby("commit")]
        oht_data = oht.fit_transform(data)
        oht_df = pd.DataFrame(oht_data, columns=oht.columns_)
        frequent_itemsets = apriori(oht_df, min_support=0.002, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        one_ante_rule = rules[[len(ante) == 1 for ante in rules["antecedants"]]]
        one_ante_rule["antecedants"] = one_ante_rule["antecedants"].apply(lambda x: next(iter(x)))
        one_ante_rule = one_ante_rule.drop("consequents", axis=1)
        max_ante_rule = one_ante_rule.groupby("antecedants").max().reset_index()

        df = df.merge(max_ante_rule, left_on="instance", right_on="antecedants")
        return df

    def get_handled_smells(self):
        return self.history_based_smells