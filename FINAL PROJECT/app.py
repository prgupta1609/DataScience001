import streamlit as st
import chromadb
from retrieval import retrieve_car_context
from generation import generate_response
import polars as pl
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Drive Wise - Automotive RAG Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


def get_unique_brands_models():
    """
    Query ChromaDB to extract unique brands and models from metadata.
    Returns sorted lists for dropdown options.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="car_brochures")
    
    # Get all data to extract unique brands and models
    results = collection.get(include=["metadatas"])
    
    if not results['metadatas']:
        return [], []
    
    # Convert to polars DataFrame for easy processing
    df = pl.DataFrame(results['metadatas'])
    
    # Get unique brands - if all are "Unknown", return "Hyundai" instead
    unique_brands = sorted(df['brand'].unique().to_list())
    
    # If all brands are "Unknown" or similar, default to "Hyundai"
    if all(brand.lower() in ["unknown", ""] for brand in unique_brands):
        unique_brands = ["Hyundai"]
    
    # Get unique models
    unique_models = sorted(df['model'].unique().to_list())
    
    return unique_brands, unique_models


def filter_models_by_brand(brand: str):
    """
    Filter models based on selected brand.
    Maps "Hyundai" to "Unknown" if that's what's stored in the database.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="car_brochures")
    
    # Map "Hyundai" to "Unknown" for database query if needed
    # Check if "Unknown" exists in the database
    all_results = collection.get(include=["metadatas"])
    if all_results['metadatas']:
        df = pl.DataFrame(all_results['metadatas'])
        stored_brands = df['brand'].unique().to_list()
        
        # If "Unknown" is stored but user selected "Hyundai", query with "Unknown"
        if brand == "Hyundai" and "Unknown" in stored_brands:
            brand = "Unknown"
    
    # Get all data with brand filter
    results = collection.get(where={"brand": brand}, include=["metadatas"])
    
    if not results['metadatas']:
        return []
    
    # Convert to polars DataFrame
    df = pl.DataFrame(results['metadatas'])
    
    # Get unique models for this brand
    unique_models = sorted(df['model'].unique().to_list())
    
    return unique_models


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar configuration
with st.sidebar:
    st.header("🚗 Vehicle Selection")
    
    # Get unique brands and models
    unique_brands, _ = get_unique_brands_models()
    
    # Brand selection
    selected_brand = st.selectbox(
        "Select Car Brand",
        options=unique_brands if unique_brands else ["No data available"],
        index=0 if unique_brands else None,
        key="brand_selector"
    )
    
    # Model selection (filtered by brand)
    if selected_brand and selected_brand != "No data available":
        available_models = filter_models_by_brand(selected_brand)
        selected_model = st.selectbox(
            "Select Model",
            options=available_models if available_models else ["No models available"],
            index=0 if available_models else None,
            key="model_selector"
        )
    else:
        selected_model = None
    
    st.divider()
    
    # API Key Status
    st.subheader("API Status")
    gemini_key = os.getenv("GEMINI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    
    st.info(f"Gemini API: {'✅ Configured' if gemini_key else '❌ Missing'}")
    st.info(f"Cohere API: {'✅ Configured' if cohere_key else '❌ Missing'}")
    
    st.divider()
    
    st.markdown("### About")
    st.markdown("""
    **Drive Wise** is a metadata-aware RAG assistant for automotive brochures.
    
    - Section-aware chunking
    - Metadata pre-filtering
    - Cohere reranking
    - Gemini generation
    """)


# Main header
st.markdown('<h1 class="main-header">Drive Wise: Metadata-Aware Automotive RAG Assistant</h1>', unsafe_allow_html=True)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if prompt := st.chat_input("Ask about vehicle specifications, features, or performance..."):
    # Validate selection
    if not selected_brand or not selected_model or selected_brand == "No data available" or selected_model == "No models available":
        st.error("Please select a valid Car Brand and Model from the sidebar.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing brochure data..."):
            try:
                # Map "Hyundai" to "Unknown" for database query if needed
                query_brand = selected_brand
                client = chromadb.PersistentClient(path="./chroma_db")
                collection = client.get_collection(name="car_brochures")
                all_results = collection.get(include=["metadatas"])
                
                if all_results['metadatas']:
                    df = pl.DataFrame(all_results['metadatas'])
                    stored_brands = df['brand'].unique().to_list()
                    
                    # If "Unknown" is stored but user selected "Hyundai", query with "Unknown"
                    if query_brand == "Hyundai" and "Unknown" in stored_brands:
                        query_brand = "Unknown"
                
                # Step 1: Retrieve context with metadata pre-filtering
                context_chunks = retrieve_car_context(
                    query=prompt,
                    brand=query_brand,
                    model=selected_model
                )
                
                if not context_chunks:
                    st.warning("No relevant information found in the brochure for this query.")
                    response = "I cannot find that specific specification in the official brochure documentation."
                    source_metadata = []
                else:
                    # Step 2: Generate response using Gemini
                    response, source_metadata = generate_response(
                        query=prompt,
                        context_chunks=context_chunks
                    )
                
                # Display the response
                st.markdown(response)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display source verification expander
                if source_metadata:
                    with st.expander("🔍 View Official Brochure Verification Sources"):
                        st.markdown("### Source Attribution")
                        
                        # Create markdown table
                        source_table = "| Section | Page Number | Relevance Score | Text Preview |\n"
                        source_table += "|---------|-------------|-----------------|--------------|\n"
                        
                        for source in source_metadata:
                            section = source.get("section", "Unknown")
                            page = source.get("page_number", "N/A")
                            score = source.get("relevance_score", 0.0)
                            text_preview = source.get("text", "")[:100] + "..." if len(source.get("text", "")) > 100 else source.get("text", "")
                            text_preview = text_preview.replace("|", "\\|")  # Escape pipe characters
                            
                            source_table += f"| {section} | {page} | {score:.4f} | {text_preview} |\n"
                        
                        st.markdown(source_table)
                        
                        # Display full text for each source
                        st.markdown("### Full Source Text")
                        for idx, source in enumerate(source_metadata, 1):
                            st.markdown(f"**Source {idx}** - {source.get('section', 'Unknown')} (Page {source.get('page_number', 'N/A')})")
                            st.text(source.get("text", ""))
                            st.divider()
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
