from fastapi import FastAPI, UploadFile, File, HTTPException, Depends,  Form
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from bson import ObjectId
from bson.json_util import dumps
import csv
import io
from dotenv import load_dotenv
import os
load_dotenv()
mongodb_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("DATABASE_NAME")
model_name = os.getenv("MODEL_NAME")

# Initialize FastAPI app
app = FastAPI()

# Load the Sentence Transformer Model
model = SentenceTransformer(model_name)

# Connect to MongoDB
client = MongoClient(mongodb_uri)
db = client[db_name]
projects_collection = db["projects"]

def bson_to_json(doc):
    """Convert BSON to JSON serializable format"""
    doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
    return doc

# Function to generate embeddings
def get_embedding(text: list):
    return model.encode(text, convert_to_tensor=False).tolist()

# Bulk add projects with embeddings
@app.post("/bulk_add_projects/")
def bulk_add_projects(  session_id: str = Form(...),  file: str = Form(...) ):
    print(session_id)
    csv_reader = csv.reader(io.StringIO(file))  # ✅ Parse CSV from string
    next(csv_reader)  # Skip header

    projects_to_insert = []
    
    for row in csv_reader:
        title, abstract = row
        embedding = get_embedding(title + abstract)
        project = {
            "title": title,
            "abstract": abstract,
            "status": "accepted",  # By default, bulk added projects are accepted
            "sessionId": ObjectId(session_id),
            "groupId": None,
            "embedding": embedding
        }
        projects_to_insert.append(project)
    print("produced all embeddings")
    projects_collection.insert_many(projects_to_insert)
    return {"message": "Projects added successfully"}

# Add a single project
@app.post("/add_project/")
def add_project(title: str, abstract: str, session_id: str, group_id: str = None):
    embedding = get_embedding([title, abstract])
    project = {
        "title": title,
        "abstract": abstract,
        "status": "pending",  # Pending by default
        "sessionId": ObjectId(session_id),
        "groupId": ObjectId(group_id) if group_id else None,
        "embedding": embedding
    }
    project_id = projects_collection.insert_one(project).inserted_id
    print("inserted all records")
    return {"projectId": str(project_id)}

# Update project title, abstract, and embedding
@app.put("/update_project/{project_id}")
def update_project(project_id: str, title: str, abstract: str):
    project = projects_collection.find_one({"_id": ObjectId(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    embedding = get_embedding([title, abstract])
    projects_collection.update_one(
        {"_id": ObjectId(project_id)},
        {"$set": {"title": title, "abstract": abstract, "embedding": embedding}}
    )
    return {"message": "Project updated successfully"}

# Get similar projects
@app.get("/get_similar_projects/{project_id}/{k}")
def get_similar_projects(project_id: str, k: int):
    project = projects_collection.find_one({"_id": ObjectId(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    query_vector = project["embedding"]
    session_id = project["sessionId"]
    
    results = projects_collection.aggregate([
        # Step 1: Perform Vector Search on filtered results
        {"$vectorSearch": {
            "queryVector": query_vector,
            "path": "embedding",
            "numCandidates": 100,
            "limit": k,
            "index": "embedding_index",
            "metric": "cosine"
        }},
        # Step 2: Filter only accepted projects from the same session
        {"$match": {"status": "accepted", "sessionId": session_id}},
        {"$project": {"_id" : 1, "title": 1,"abstract": 1, "cosineSimilarity": {"$meta": "vectorSearchScore"}  }}
    ])
    return [bson_to_json(doc) for doc in results]
