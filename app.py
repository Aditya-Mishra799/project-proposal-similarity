from fastapi import FastAPI, UploadFile, File, HTTPException, Depends,  Form
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from bson import ObjectId
from bson.json_util import dumps
import csv
import io
from dotenv import load_dotenv
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class ProjectCreateRequest(BaseModel):
    title: str
    abstract: str
    session_id: str
    creator_id: str


load_dotenv()
mongodb_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("DATABASE_NAME")
model_name = os.getenv("MODEL_NAME")

# Initialize FastAPI app
app = FastAPI()

# Allow CORS from specific origins
origins = [
    "http://localhost:3000",  # Example: Allowing localhost with port 3000
    "https://myfrontenddomain.com",  # Example: A specific frontend domain
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    # List of allowed origins (you can use '*' to allow all)
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Allowed HTTP methods
    allow_headers=["X-Custom-Header", "Content-Type"],  # Allowed headers
)

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
async def bulk_add_projects(session_id: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()  # Read file content
    csv_reader = csv.reader(io.StringIO(
        contents.decode("utf-8")))  # Convert to string
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
            "creator": None,
            "embedding": embedding,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
        }
        projects_to_insert.append(project)
    print("produced all embeddings")
    projects_collection.insert_many(projects_to_insert)
    return {"message": "Projects added successfully"}

# Add a single project


@app.post("/add_project/")
def add_project(project: ProjectCreateRequest):
    title, abstract, session_id, creator_id = (
        project.title,
        project.abstract,
        project.session_id,
        project.creator_id
    )
    embedding = get_embedding(title + abstract)
    project = {
        "title": title,
        "abstract": abstract,
        "status": "pending",  # Pending by default
        "sessionId": ObjectId(session_id),
        "embedding": embedding,
        "creator": ObjectId(creator_id),
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow(),
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

    embedding = get_embedding(title + abstract)
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
    print(query_vector[0])

    results = projects_collection.aggregate([
        # Step 1: Perform Vector Search on filtered results
        {"$vectorSearch": {
            "queryVector": query_vector,
            "path": "embedding",
            "numCandidates": 100,
            "limit": k,
            "index": "embedding_index",
            "metric": "cosine",
            "filter": {
                "$and": [
                    {"status": "accepted"},
                    {"sessionId": ObjectId(session_id)},
                    {"_id": {"$ne": ObjectId(project_id)}},
                ]

            }
        }},
        {"$project": {"_id": 1, "title": 1, "abstract": 1,
                      "cosineSimilarity": {"$meta": "vectorSearchScore"}}}
    ])
    return [bson_to_json(doc) for doc in results]
