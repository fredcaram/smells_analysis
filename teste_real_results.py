from pathlib import Path
import os
import re
import numpy as np
import pandas as pd

identified_path = os.path.join(".",
                               "original_experiment_dataset",
                               "Replication Package",
                               "Study I - Identified Smells",
                               "ParallelInheritance")
pathlist = Path(identified_path).glob('*HIST.csv')
identified_values = {}
for path in pathlist:
    filepath = str(path)
    f_name = os.path.basename(filepath)
    m = re.match('^(\w+)-.*$', f_name)
    system_name = m.group(1)
    with open(filepath) as f:
        values = f.readlines()

        identified_values[system_name] = [x.strip() for x in values if len(x.strip()) > 0]

oracle_path = os.path.join(".",
                           "original_experiment_dataset",
                           "Replication Package",
                           "Study I - Oracles",
                           "ParallelInheritance")
pathlist = Path(oracle_path).glob('*-ORACLE.csv')
oracle_values = {}
for path in pathlist:
    filepath = str(path)
    f_name = os.path.basename(filepath)
    m = re.match('^(\w+)-.*$', f_name)
    system_name = m.group(1)
    with open(filepath) as f:
        values = f.readlines()

        oracle_values[system_name] = [x.strip() for x in values if len(x.strip()) > 0]

total_tp = 0
total_fp = 0
total_fn = 0

keys = np.unique(np.concatenate((
    list(identified_values.keys()),
    list(oracle_values.keys()))))
for key in keys:
    tp=0
    fp=0
    fn=0
    identified = np.array(identified_values[key]) if key in identified_values else np.array([])
    oracle = np.array(oracle_values[key]) if key in oracle_values else np.array([])

    tp = np.sum(np.vectorize(lambda i: i in oracle)(identified)) if len(identified) > 0 else 0
    fp = np.sum(np.vectorize(lambda i: i not in oracle)(identified)) if len(identified) > 0 else 0
    fn = np.sum(np.vectorize(lambda i: i not in identified)(oracle)) if len(oracle) > 0 else 0

    total_tp += tp
    total_fp += fp
    total_fn += fn

    precision = (tp / len(identified)) if len(identified) > 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    fmeasure = (2 / (1/recall + 1/precision)) if recall != 0 and precision != 0 else 0
    print("Results for {0}".format(key))
    print("tp: {0}   fp: {1}     fn: {2}".format(tp, fp, fn))
    print("precision: {0}   recall: {1}     f-measure: {2}".format(precision, recall, fmeasure))

precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0
recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0
fmeasure = (2 / (1/recall + 1/precision)) if recall != 0 and precision != 0 else 0
print("Total Results")
print("tp: {0}   fp: {1}     fn: {2}".format(total_tp, total_fp, total_fn))
print("precision: {0}   recall: {1}     f-measure: {2}".format(precision, recall, fmeasure))