import json
from handle_landfill_data.http_helper import http_helper
from handle_landfill_data.mongodb_helper import mongodb_helper

dataSetId = 2

systems_url = "http://www.sesa.unisa.it/landfill/GetSystems?datasetId=" + str(dataSetId)
smells_url = "http://www.sesa.unisa.it/landfill/GetBadSmells?system={0}&type={1}"
httphelper = http_helper()
mongohelper = mongodb_helper()

smells_projects_collection = mongohelper.get_collection("smells_projects")
smells_projects_collection.delete_many({"dataset_id": dataSetId})
smells_collection = mongohelper.get_collection("smells")
smells_collection.delete_many({"dataset_id": dataSetId})

systemsJson = json.loads(httphelper.http_get_data(systems_url))
for item in systemsJson:
    item["dataset_id"] = dataSetId
mongohelper.insert_many("smells_projects", systemsJson)
smells_db = mongohelper.get_db()

for project in smells_db.smells_projects.find():
    project_id = project["id"]
    smells = project["types"]
    for smell in smells:
        smell_url = smells_url.format(project_id, smell["type"])
        resp = httphelper.http_get_data(smell_url)
        if len(resp) > 0:
            smellsJson = json.loads(resp)
            smellsData = smellsJson["data"]
            for smellData in smellsData:
                smellData["smells_projects_id"] = project["_id"]
                smellData["project_id"] = project["id"]
                smellData["dataset_id"] = dataSetId
                mongohelper.insert_one("smells", smellData)

mongohelper.set_projects_prefix()
print(systemsJson)
