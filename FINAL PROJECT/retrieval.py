import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_cohere import CohereEmbeddings
from cohere import Client as CohereClient
from dotenv import load_dotenv
import polars as pl
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def retrieve_car_context(query: str, brand: str, model: str) -> List[Dict[str, Any]]:
    """
    Retrieve relevant car context using metadata pre-filtering and Cohere reranking.
    
    Args:
        query: User's search query
        brand: Car brand (e.g., "Hyundai")
        model: Car model (e.g., "Ioniq5")
    
    Returns:
        List of top 3 re-ranked results with metadata
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="car_brochures")
    
    # Initialize Cohere embeddings (must match ingestion)
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    # Apply strict metadata pre-filter
    metadata_filter = {
        "$and": [
            {"brand": brand},
            {"model": model}
        ]
    }
    
    logger.info(f"Retrieving context for {brand} {model} with query: '{query}'")
    
    # Generate query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Query ChromaDB with metadata filter and retrieve top 12 results
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=12,
        where=metadata_filter
    )
    
    # Extract results
    if not results['ids'][0]:
        logger.warning(f"No results found for {brand} {model}")
        return []
    
    # Prepare telemetry tracking with polars
    telemetry_data = []
    
    # Build context list for reranking
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    # Track initial retrieval
    telemetry_data.append({
        "stage": "initial_retrieval",
        "brand": brand,
        "model": model,
        "query": query,
        "results_count": len(documents)
    })
    
    # Initialize Cohere Rerank client
    cohere_client = CohereClient(os.getenv("COHERE_API_KEY"))
    
    # Rerank using Cohere API
    rerank_results = cohere_client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=3
    )
    
    # Extract top 3 re-ranked results with metadata
    final_results = []
    
    for idx, rerank_result in enumerate(rerank_results.results):
        original_index = rerank_result.index
        relevance_score = rerank_result.relevance_score
        
        # Get corresponding metadata
        metadata = metadatas[original_index]
        
        result = {
            "text": documents[original_index],
            "section": metadata.get("section", "Unknown"),
            "page_number": metadata.get("page_number", 0),
            "brand": metadata.get("brand", brand),
            "model": metadata.get("model", model),
            "document_version": metadata.get("document_version", "v1.0"),
            "relevance_score": relevance_score,
            "rank": idx + 1
        }
        
        final_results.append(result)
        
        # Track reranking telemetry
        telemetry_data.append({
            "stage": "reranking",
            "brand": brand,
            "model": model,
            "original_index": original_index,
            "rerank_score": relevance_score,
            "final_rank": idx + 1,
            "section": metadata.get("section", "Unknown")
        })
    
    # Convert telemetry to polars DataFrame for structured tracking
    telemetry_df = pl.DataFrame(telemetry_data)
    logger.info(f"Telemetry tracking completed:\n{telemetry_df}")
    
    logger.info(f"Retrieved {len(final_results)} re-ranked results for {brand} {model}")
    
    return final_results


if __name__ == "__main__":
    # Example usage
    test_query = "What is the fuel efficiency and mileage?"
    test_brand = "Hyundai"
    test_model = "Ioniq5"
    
    # Validate API key
    if not os.getenv("COHERE_API_KEY") or os.getenv("COHERE_API_KEY") == "your_cohere_api_key_here":
        logger.error("COHERE_API_KEY not found or not set in environment variables.")
        exit(1)
    
    # Test retrieval
    results = retrieve_car_context(test_query, test_brand, test_model)
    
    # Print results
    print("\n" + "="*80)
    print("RETRIEVAL RESULTS")
    print("="*80)
    for idx, result in enumerate(results, 1):
        print(f"\nResult #{idx} (Score: {result['relevance_score']:.4f})")
        print(f"Section: {result['section']}")
        print(f"Page: {result['page_number']}")
        print(f"Text: {result['text'][:200]}...")
    print("="*80)
