import chromadb
from chromadb.config import Settings
import os

def initialize_database():
    """
    Initialize a local persistent ChromaDB client and create the car_brochures collection.
    """
    # Create the chroma_db directory if it doesn't exist
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=db_path)
    
    # Check if the collection already exists
    collection_name = "car_brochures"
    existing_collections = [col.name for col in client.list_collections()]
    
    if collection_name in existing_collections:
        print(f"Collection '{collection_name}' already exists. Skipping creation.")
        return
    
    # Create the collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Car brochures and automotive documentation"}
    )
    
    print(f"Successfully created collection '{collection_name}' in ChromaDB.")
    print(f"Database is stored at: {db_path}")

if __name__ == "__main__":
    initialize_database()
