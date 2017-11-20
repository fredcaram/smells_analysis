from pymongo import MongoClient

class mongodb_helper:
    def get_db(self):
        client = MongoClient('localhost', 27017)
        return client.smells_dataset

    def insert_one(self, collection_name, collection_values):
        self.get_collection(collection_name).insert_one(collection_values)

    def insert_many(self, collection_name, collection_values):
        self.get_collection(collection_name).insert_many(collection_values)

    def drop_collection(self, collection_name):
        self.get_collection(collection_name).drop()

    def get_collection(self, collection_name):
        db = self.get_db()
        return db[collection_name]

    def set_projects_prefix(self):
        project_prefix_dict = [
            {
                "id" : 55,
                "prefix" : "aard"
            },
            {
                "id" : 71,
                "prefix" : "andengine"
            },
            {
                "id" : 52,
                "prefix" : "android_base"
            },
            {
                "id" : 80,
                "prefix" : "android_tel"
            },
            {
                "id" : 64,
                "prefix" : "android_sdk"
            },
            {
                "id" : 81,
                "prefix" : "android_sup"
            },
            {
                "id" : 86,
                "prefix" : "android_tool_base"
            },
            {
                "id" : 54,
                "prefix" : "ant"
            },
            {
                "id" : 63,
                "prefix" : "cassandra"
            },
            {
                "id" : 78,
                "prefix" : "apache_codec"
            },
            {
                "id" : 72,
                "prefix" : "apache_io"
            },
            {
                "id" : 60,
                "prefix" : "apache_lang"
            },
            {
                "id" : 56,
                "prefix" : "apache_logging"
            },
            {
                "id" : 79,
                "prefix" : "apache_derby"
            },
            {
                "id" : 73,
                "prefix" : "apache_james"
            },
            {
                "id" : 70,
                "prefix" : "apache_tomcat"
            },
            {
                "id" : 61,
                "prefix" : "eclipse_core"
            },
            {
                "id" : 77,
                "prefix" : "google_guava"
            },
            {
                "id" : 49,
                "prefix" : "jedit"
            },
            {
                "id" : 57,
                "prefix" : "mongo_java"
            }
        ]
        coll = self.get_collection("smells_projects")
        for project_prefix_item in project_prefix_dict:
            coll.update_one(({"id": project_prefix_item["id"]}, {"$set": {"prefix": project_prefix_item["prefix"]}}))