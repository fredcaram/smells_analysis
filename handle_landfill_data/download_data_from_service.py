import json
from handle_landfill_data.http_helper import http_helper
from handle_landfill_data.mongodb_helper import mongodb_helper

dataSetId = 2

systems_url = "http://www.sesa.unisa.it/landfill/GetSystems?datasetId=" + str(dataSetId)
smells_url = "http://www.sesa.unisa.it/landfill/GetBadSmells?system={0}&type={1}"
httphelper = http_helper()
mongohelper = mongodb_helper()

mongohelper.drop_collection("smells_projects")
mongohelper.drop_collection("smells")

systemsJson = json.loads(httphelper.http_get_data(systems_url))
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
                mongohelper.insert_one("smells", smellData)

print(systemsJson)
