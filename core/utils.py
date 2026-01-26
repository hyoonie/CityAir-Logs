import pymongo
from CityAirLogs import settings

def get_db_collection():
    db_config = settings.MONGO_SETTINGS

    client = pymongo.MongoClient(
        host=db_config["HOST"],
        port=db_config["PORT"],
        username=db_config["USERNAME"],
        password=db_config["PASSWORD"],
        authSource=db_config["AUTH_SOURCE"],
        authMechanism=db_config.get("AUTH_MECHANISM", "SCRAM-SHA-1"),
    )

    db = client[db_config["DB_NAME"]]

    return client, db["lecturas"]


def get_distinct_devices():
    """Obtiene lista única de dispositivos para el checkbox"""
    client, collection = get_db_collection()
    devices = collection.distinct("idDispositivo")
    client.close()
    return sorted([d for d in devices if d])


def get_mongo_data(query_filter):
    """Ejecuta la consulta directamente en Mongo"""
    client, collection = get_db_collection()
    data = list(collection.find(query_filter, {"_id": 0}))
    client.close()
    return data