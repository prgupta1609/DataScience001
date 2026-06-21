# Unstructured Data QA System (Scratch RAG Pipeline)

This project implements a complete, modular Retrieval-Augmented Generation (RAG) pipeline designed for question-answering over unstructured custom texts or domain-specific data archives without external API tool dependencies.

## 🏗️ System Architecture & Framework
The system is built sequentially across 8 core evaluation guidelines:
1. **Adaptive Ingestion**: Loads mock text archives matching open_ragbench benchmark structure natively in memory.
2. **Token Window Chunking**: Partitions text into distinct chunks using a fixed sliding window strategy.
3. **Semantic Embeddings**: Implements word coordinates mapping with uniform normalization bounded between 0 and 1.
4. **Vector Database**: Indexes continuous text structures natively for fast proximity calculations.
5. **Query Processing**: Transforms incoming text questions to vector space layout variables.
6. **Proximity Search Engine**: Executes similarity matching routines based on standard Cosine Proximity formulas.
7. **Grounded Prompt Assembly**: Bounds language prompts strictly within context domains to eliminate hallucinations.
8. **System Optimization Analysis**: Evaluates parameters by generating side-by-side comparative metric trajectories.

## 📊 Core Mathematical Constraint
Cosine Similarity between user query ($\mathbf{q}$) and document chunks ($\mathbf{v}$) is evaluated using:

$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{v}) = \frac{\mathbf{q} \cdot \mathbf{v}}{\|\mathbf{q}\| \|\mathbf{v}\|}$$

Strict L2 normalization guarantees vectors remain within standard positive unit bounds ($\|\mathbf{v}\| = 1$), avoiding negative trajectory divergence on tracking charts.

## 🚀 Execution Guide
1. Install core baseline deep learning utilities: `pip install -r requirements.txt`
2. Open the executed workspace: `week7_PriyaGupta.ipynb`
3. Click **"Clear All Outputs"** followed by **"Restart Kernel"**.
4. Run all execution nodes sequentially to render side-by-side baseline vs. optimized performance graphs.