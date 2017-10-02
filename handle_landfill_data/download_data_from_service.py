import json
from handle_landfill_data.http_helper import http_helper

dataSetId = 2

systems_url = "http://www.sesa.unisa.it/landfill/GetSystems?datasetId=" + str(dataSetId)
httphelper = http_helper()

systemsJson = json.loads(httphelper.http_get_data(systems_url))
projects = {}
smells = {}
for project in systemsJson:
    project_smells = project["types"]
    project_name = project["name"]
    project_id = project["id"]
    project_snapshot = project["snapshot"]


print(systemsJson)
