import os
from typing import List, Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import polars as pl
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# System prompt with strict guardrails
SYSTEM_PROMPT = """
You are Drive Wise, an expert automotive assistant designed to help users understand car brochures clearly.
Your core directive is to answer the user's question using ONLY the explicit context snippets provided below. 

Strict Constraints:
If the provided text chunks do not contain a direct, definitive answer to the question, reply verbatim: "I cannot find that specific specification in the official brochure documentation." Do not extrapolate, make assumptions, or introduce outside general training knowledge.
Format all engineering specs, performance metrics, and equipment lists into highly readable markdown tables or bullet points.
Never cross-reference or mention features belonging to other car models or brands.
"""


def generate_response(query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate a response using Gemini 2.5 Flash with strict guardrails.
    
    Args:
        query: User's question
        context_chunks: List of re-ranked context chunks with metadata
    
    Returns:
        Tuple of (generated_answer, source_metadata_list)
    """
    # Initialize Gemini LLM with deterministic settings
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0,  # Force rigid, factual execution
    )
    
    # Extract context text and metadata
    context_text = ""
    source_metadata = []
    
    telemetry_data = []
    
    for idx, chunk in enumerate(context_chunks):
        # Build context text with section headers
        context_text += f"\n--- Context Chunk {idx + 1} (Section: {chunk.get('section', 'Unknown')}, Page: {chunk.get('page_number', 'N/A')}) ---\n"
        context_text += chunk['text'] + "\n"
        
        # Collect source metadata for citation tracking
        source_metadata.append({
            "section": chunk.get("section", "Unknown"),
            "page_number": chunk.get("page_number", 0),
            "brand": chunk.get("brand", "Unknown"),
            "model": chunk.get("model", "Unknown"),
            "relevance_score": chunk.get("relevance_score", 0.0),
            "rank": idx + 1
        })
        
        # Track generation telemetry
        telemetry_data.append({
            "stage": "context_assembly",
            "chunk_index": idx,
            "section": chunk.get("section", "Unknown"),
            "page_number": chunk.get("page_number", 0),
            "relevance_score": chunk.get("relevance_score", 0.0)
        })
    
    # Construct the full prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\nContext Snippets:\n{context_text}\n\nUser Question: {query}"
    
    logger.info(f"Generating response for query: '{query}'")
    logger.info(f"Using {len(context_chunks)} context chunks")
    
    try:
        # Generate response
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context Snippets:\n{context_text}\n\nUser Question: {query}")
        ]
        
        response = llm.invoke(messages)
        generated_answer = response.content
        
        # Track generation telemetry
        telemetry_data.append({
            "stage": "generation",
            "query": query,
            "context_chunks_used": len(context_chunks),
            "response_length": len(generated_answer),
            "model": "gemini-2.5-flash",
            "temperature": 0.0
        })
        
        # Convert telemetry to polars DataFrame
        telemetry_df = pl.DataFrame(telemetry_data)
        logger.info(f"Generation telemetry:\n{telemetry_df}")
        
        logger.info("Response generated successfully")
        
        return generated_answer, source_metadata
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        
        # Track error telemetry
        telemetry_data.append({
            "stage": "generation_error",
            "query": query,
            "error_message": str(e)
        })
        
        error_telemetry_df = pl.DataFrame(telemetry_data)
        logger.error(f"Error telemetry:\n{error_telemetry_df}")
        
        # Return error response
        error_answer = "I apologize, but I encountered an error while generating the response. Please try again."
        return error_answer, source_metadata


if __name__ == "__main__":
    # Example usage
    test_query = "What is the fuel efficiency and mileage?"
    
    # Mock context chunks (in production, these come from retrieval.py)
    test_context_chunks = [
        {
            "text": "The Hyundai Ioniq 5 delivers an impressive driving range of up to 507 km on a single charge, with a fuel efficiency equivalent of 138 MPGe.",
            "section": "Mileage & Fuel Efficiency",
            "page_number": 12,
            "brand": "Hyundai",
            "model": "Ioniq5",
            "relevance_score": 0.95
        },
        {
            "text": "The vehicle features a 72.6 kWh battery pack with ultra-fast charging capability, reaching 10% to 80% charge in just 18 minutes.",
            "section": "Engine & Performance",
            "page_number": 8,
            "brand": "Hyundai",
            "model": "Ioniq5",
            "relevance_score": 0.87
        }
    ]
    
    # Validate API key
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not found in environment variables.")
        exit(1)
    
    # Test generation
    answer, metadata = generate_response(test_query, test_context_chunks)
    
    # Print results
    print("\n" + "="*80)
    print("GENERATION RESULTS")
    print("="*80)
    print(f"\nQuery: {test_query}")
    print(f"\nGenerated Answer:\n{answer}")
    print(f"\nSource Metadata:")
    for idx, meta in enumerate(metadata, 1):
        print(f"  Source #{idx}: Section={meta['section']}, Page={meta['page_number']}, Score={meta['relevance_score']:.4f}")
    print("="*80)
