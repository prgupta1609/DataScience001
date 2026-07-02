import os
import re
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Automotive section headers for intelligent chunking
SECTION_HEADERS = [
    r'engine\s*&\s*performance',
    r'mileage\s*&\s*fuel\s*efficiency',
    r'safety',
    r'dimensions',
    r'interior\s*&\s*comfort',
    r'infotainment\s*&\s*connectivity',
    r'exterior',
    r'transmission',
    r'suspension',
    r'brakes',
    r'features',
    r'specifications',
    r'technical\s*specs',
    r'overview',
    r'introduction'
]

# Compile regex pattern for section detection
SECTION_PATTERN = re.compile('|'.join(SECTION_HEADERS), re.IGNORECASE)


def extract_brand_model_from_filename(filename: str) -> tuple[str, str]:
    """
    Extract brand and model from filename.
    Expected format: Brand_Model.pdf or Brand_Model_Version.pdf
    """
    # Remove extension and split by underscore
    name_without_ext = Path(filename).stem
    parts = name_without_ext.split('_')
    
    if len(parts) >= 2:
        brand = parts[0].strip()
        model = '_'.join(parts[1:]).strip()
        return brand, model
    else:
        # Fallback if filename doesn't match expected pattern
        return "Unknown", name_without_ext


def section_aware_chunking(text: str, page_num: int) -> List[Dict[str, Any]]:
    """
    Split text into chunks based on section headers.
    Returns list of chunks with metadata.
    """
    chunks = []
    
    # Split by section headers while preserving the header
    lines = text.split('\n')
    current_section = "General"
    current_chunk_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line is a section header
        section_match = SECTION_PATTERN.search(line)
        if section_match:
            # Save previous chunk if exists
            if current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'section': current_section,
                        'page_number': page_num
                    })
            
            # Start new chunk with new section
            current_section = section_match.group(0).title()
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)
    
    # Don't forget the last chunk
    if current_chunk_lines:
        chunk_text = '\n'.join(current_chunk_lines).strip()
        if chunk_text:
            chunks.append({
                'text': chunk_text,
                'section': current_section,
                'page_number': page_num
            })
    
    return chunks


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using pypdf with section-aware chunking.
    Returns list of chunks with metadata.
    """
    try:
        import pypdf
    except ImportError:
        logger.error("pypdf not installed. Please install it first.")
        return []
    
    chunks = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        for page_num in range(total_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            
            if text:
                page_chunks = section_aware_chunking(text, page_num + 1)
                chunks.extend(page_chunks)
                
        logger.info(f"Extracted {len(chunks)} chunks from {total_pages} pages")
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
    
    return chunks


def process_documents(dataset_dir: str, chroma_db_path: str, collection_name: str):
    """
    Process all PDFs in dataset directory and ingest into ChromaDB.
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=chroma_db_path)
    
    # Get or create collection
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"Connected to existing collection '{collection_name}'")
    except:
        collection = client.create_collection(name=collection_name)
        logger.info(f"Created new collection '{collection_name}'")
    
    # Initialize Cohere embeddings
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    # Process all PDF files
    dataset_path = Path(dataset_dir)
    pdf_files = list(dataset_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {dataset_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    total_chunks_processed = 0
    document_summary = {}
    
    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")
        
        # Extract brand and model from filename
        brand, model = extract_brand_model_from_filename(pdf_file.name)
        document_version = "v1.0"  # Default version, can be enhanced
        
        # Extract text chunks with section-aware chunking
        chunks = extract_text_from_pdf(str(pdf_file))
        
        if not chunks:
            logger.warning(f"No chunks extracted from {pdf_file.name}")
            continue
        
        # Prepare batch data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{brand}_{model}_page{chunk['page_number']}_chunk{idx}"
            ids.append(chunk_id)
            documents.append(chunk['text'])
            metadatas.append({
                'brand': brand,
                'model': model,
                'section': chunk['section'],
                'page_number': chunk['page_number'],
                'document_version': document_version,
                'source_file': pdf_file.name
            })
        
        # Generate embeddings
        try:
            embedding_vectors = embeddings.embed_documents(documents)
            
            # Batch upsert to ChromaDB
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedding_vectors
            )
            
            chunk_count = len(chunks)
            total_chunks_processed += chunk_count
            document_summary[pdf_file.name] = chunk_count
            
            logger.info(f"✓ Ingested {chunk_count} chunks from {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Error embedding/ingesting {pdf_file.name}: {str(e)}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INGESTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total documents processed: {len(pdf_files)}")
    logger.info(f"Total chunks ingested: {total_chunks_processed}")
    logger.info("\nChunks per document:")
    for doc_name, count in document_summary.items():
        logger.info(f"  {doc_name}: {count} chunks")
    logger.info("="*60)


if __name__ == "__main__":
    # Configuration
    DATASET_DIR = "./dataset"
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "car_brochures"
    
    # Validate COHERE_API_KEY
    if not os.getenv("COHERE_API_KEY") or os.getenv("COHERE_API_KEY") == "your_cohere_api_key_here":
        logger.error("COHERE_API_KEY not found or not set in environment variables. Please set it in .env file.")
        exit(1)
    
    # Process documents
    process_documents(DATASET_DIR, CHROMA_DB_PATH, COLLECTION_NAME)
