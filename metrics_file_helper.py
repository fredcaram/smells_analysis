import re
from pathlib import Path

#file_suffixes= ["dependencies", "complexity"]
raw_metrics_folder = "raw_metrics_files"
metrics_folder = "metrics_files"
datasets_folders = ["dataset_2"]

def get_project_prefix(file_name):
    regex_str = '(\w*)_.*\.csv'
    regex_match = re.match(regex_str, file_name)
    return regex_match.group(1)

def handle_raw_metrics_files():
    for ds_folder in datasets_folders:
        pathlist = Path("{0}/{1}".format(raw_metrics_folder, ds_folder)).glob('**/*.csv')
        for path in pathlist:
            filepath = str(path)
            project_prefix = get_project_prefix(path.name)
            print(filepath)
            print(project_prefix)
            split_files_by_type(filepath, project_prefix, ds_folder)


def split_files_by_type(csv_file_path, project_prefix, ds_folder):
    lineNumber = 1
    newFile = True
    for line in open(csv_file_path, "r").readlines():
        if lineNumber == 1:
            lineNumber += 1
            continue
        if newFile:
            instance_type = get_instance_type(line)
            outHandle = open("./{0}/{1}/{2}_{3}.csv".format(metrics_folder, ds_folder, project_prefix, instance_type.lower()), "w")
            newFile = False
        outHandle.write(line)
        if not line.strip():
            if outHandle is not None:
                outHandle.close()
            newFile = True
        lineNumber += 1
    outHandle.close()

def get_instance_type(line):
    regex_match = re.match("^(\w+),.*", line)
    instance_type = regex_match.group(1)
    return instance_type

handle_raw_metrics_files()