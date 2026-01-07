import json
import numpy as np
import os
from typing import List

# LlamaIndex imports
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def stream_documents_from_jsonl(file_paths: List[str]):
    """
    Generator that yields LlamaIndex Document objects from JSONL files.
    """
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Create a LlamaIndex Document if the text field exists
                    if "query" in data:
                        yield Document(text=data["query"], metadata=data)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at {file_path}:{line_num}")

def create_embeddings_with_llama_index(
    file_paths: List[str],
    model_name: str = "facebook/contriever",
    batch_size: int = 512,
) -> np.ndarray:
    """
    Uses LlamaIndex to generate embeddings from JSONL files.
    """
    print(f"Initializing LlamaIndex embedding model: {model_name}...")
    
    # Initialize the embedding model
    # device='auto' will automatically pick CUDA or MPS (Apple Silicon) if available
    embed_model = HuggingFaceEmbedding(model_name=model_name, device="cuda")

    all_embeddings = []
    current_batch_texts = []

    print("Processing documents...")
    
    # Stream Documents and process in batches
    for doc in stream_documents_from_jsonl(file_paths):
        current_batch_texts.append(doc.text)

        if len(current_batch_texts) >= batch_size:
            # get_text_embedding_batch returns a list of lists [ [float, ...], [float, ...] ]
            batch_embeddings = embed_model.get_text_embedding_batch(current_batch_texts)
            all_embeddings.append(np.array(batch_embeddings))
            current_batch_texts = []
            total_emb = sum(len(b) for b in all_embeddings)
            print(f"Processed {sum(len(b) for b in all_embeddings)} vectors...", end='\r')
            if total_emb >= 1000000:
                break

    # Process remaining items
    # if current_batch_texts:
    #     batch_embeddings = embed_model.get_text_embedding_batch(current_batch_texts)
    #     all_embeddings.append(np.array(batch_embeddings))

    if not all_embeddings:
        return np.array([])

    # Stack into a single (N, Dim) numpy array
    final_array = np.vstack(all_embeddings)
    np.save("wiki_query.npy", final_array)
    
    return final_array

if __name__ == "__main__":
    # --- Configuration ---
    my_files = [
        "/mnt/local/yongye/LEANN/benchmarks/data/queries/nq_open.jsonl",
        # "data/part2.jsonl"
    ]
    
    # --- Run ---
    if not my_files:
        print("Please add files to the 'my_files' list.")
    else:
        # Note: You need to install llama-index and llama-index-embeddings-huggingface
        # pip install llama-index llama-index-embeddings-huggingface
        embeddings = create_embeddings_with_llama_index(my_files)
        
        # Optional: Save
        # np.save("llama_embeddings.npy", embeddings)