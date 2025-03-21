from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.admin.command('ismaster')
    print("MongoDB connection successful")
except ConnectionFailure:
    print("MongoDB server is not available")
    # Handle the error (e.g., exit the script or use a fallback database)