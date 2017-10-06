from pymongo import MongoClient

class mongodb_helper:
    def get_db(self):
        client = MongoClient('localhost', 27017)
        return client.smells_dataset

    def insert_one(self, collection_name, collection_values):
        db = self.get_db()
        db[collection_name].insert_one(collection_values)

    def insert_many(self, collection_name, collection_values):
        db = self.get_db()
        db[collection_name].insert_many(collection_values)

    def drop_collection(self, collection_name):
        db = self.get_db()
        db[collection_name].drop()

    def set_projects_prefix(self):
        db = self.get_db()
        project_prefix_dict = {55: "aard"}
        coll = db["smells_projects"]
        for id, prefix in project_prefix_dict.items():
            coll.update_one(({"id": id}, {"$set": {"prefix", prefix}}))