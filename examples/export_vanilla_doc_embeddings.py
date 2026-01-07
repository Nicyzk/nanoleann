#!/usr/bin/env python3
"""Test only Faiss HNSW"""

import os
import sys
import time

import psutil

import numpy as np


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class MemoryTracker:
    def __init__(self, name: str):
        self.name = name
        self.start_mem = get_memory_usage()
        self.stages = []

    def checkpoint(self, stage: str):
        current_mem = get_memory_usage()
        diff = current_mem - self.start_mem
        print(f"[{self.name} - {stage}] Memory: {current_mem:.1f} MB (+{diff:.1f} MB)")
        self.stages.append((stage, current_mem))
        return current_mem

    def summary(self):
        peak_mem = max(mem for _, mem in self.stages)
        print(f"Peak Memory: {peak_mem:.1f} MB")
        return peak_mem


def main():
    import os
    from llama_index.core import SimpleDirectoryReader, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # 1. Setup your embedding model
    # (Using the path from your previous context)
    embed_model_path = "/mnt/local/yongye/hub/models--facebook--contriever/snapshots/2bd46a25019aeea091fd42d1f0fd4801675cf699"
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_path)

    # 2. Load Documents
    print("Loading documents...")
    documents = SimpleDirectoryReader("doc_data", recursive=True).load_data()
    print(f"Loaded {len(documents)} source files.")

    # 3. Parse Documents into Nodes (Chunks)
    # This is crucial. You cannot embed a whole PDF at once.
    parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Split documents into {len(nodes)} chunks (nodes).")

    # 4. Generate Embeddings
    # We extract the text content from all nodes and batch process them
    print("Generating embeddings (this may take a moment)...")

    # Retrieve just the text content for embedding
    texts_to_embed = [n.get_content(metadata_mode="embed") for n in nodes]

    # Run the model
    embeddings = Settings.embed_model.get_text_embedding_batch(
        texts_to_embed, 
        show_progress=True
    )

    # 5. Attach embeddings back to the nodes (Optional but recommended)
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding

    # --- RESULT ---
    print(f"\nSuccess! Generated {len(embeddings)} vectors.")
    print(f"First vector dimension: {len(embeddings[0])}")
    print(f"First vector sample: {embeddings[0][:5]}")

    np_emb = np.array(embeddings, dtype=np.float16)
    print(np_emb.shape)
    # import pdb; pdb.set_trace()
    np.save("all_embeddings.npy", np_emb)

    # You now have two lists you can use:
    # 1. 'embeddings': A list of lists containing the raw float vectors
    # 2. 'nodes': The metadata and text corresponding to those vectors
    return 

    tracker.checkpoint("After text splitter setup")

    # Check if index already exists and try to load it
    index_loaded = False
    if os.path.exists("./storage_faiss"):
        print("Loading existing Faiss HNSW index...")
        try:
            # Use the correct Faiss loading pattern from the example
            vector_store = FaissVectorStore.from_persist_dir("./storage_faiss")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir="./storage_faiss"
            )
            from llama_index.core import load_index_from_storage

            index = load_index_from_storage(storage_context=storage_context)
            print("Index loaded from ./storage_faiss")
            tracker.checkpoint("After loading existing index")
            index_loaded = True
        except Exception as e:
            print(f"Failed to load existing index: {e}")
            print("Cleaning up corrupted index and building new one...")
            # Clean up corrupted index
            import shutil

            if os.path.exists("./storage_faiss"):
                shutil.rmtree("./storage_faiss")

    if not index_loaded:
        print("Building new Faiss HNSW index...")

        # Use the correct Faiss building pattern from the example
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, transformations=[node_parser]
        )
        tracker.checkpoint("After index building")

        # Save index to disk using the correct pattern
        index.storage_context.persist(persist_dir="./storage_faiss")
        tracker.checkpoint("After index saving")

    # Measure runtime memory overhead
    print("\nMeasuring runtime memory overhead...")
    runtime_start_mem = get_memory_usage()
    print(f"Before load memory: {runtime_start_mem:.1f} MB")
    tracker.checkpoint("Before load memory")

    # ... inside main(), after memory measurement setup ...

    # REPLACE THIS SECTION:
    # query_engine = index.as_query_engine(similarity_top_k=20)
    # ... loop using query_engine.query() ...

    # WITH THIS:
    print("Initializing Retriever (skipping LLM synthesis)...")
    retriever = index.as_retriever(similarity_top_k=20)
    
    queries = [
        "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面,任务令一般在什么城市颁发",
        "What is LEANN and how does it work?",
        "华为诺亚方舟实验室的主要研究内容",
    ]

    for i, query in enumerate(queries):
        start_time = time.time()
        
        # .retrieve() runs the embedding + faiss search ONLY.
        # It returns a list of Node objects.
        nodes = retriever.retrieve(query)
        
        query_time = time.time() - start_time
        print(f"Query {i + 1} time: {query_time:.3f}s")
        tracker.checkpoint(f"After query {i + 1}")
        
        print(f"Query: {query}")
        for i, node_with_score in enumerate(nodes):
            # 1. Access the similarity score
            score = node_with_score.score
            
            # 2. Access the actual underlying Node object
            real_node = node_with_score.node
            
            # 3. Get content and metadata for verification
            content = real_node.get_content()  # The text chunk
            metadata = real_node.metadata      # Dictionary with 'file_name', 'page_label', etc.
            node_id = real_node.node_id        # Unique ID of the chunk
            
            print(node_id)

    # ... rest of memory calculation ...

    runtime_end_mem = get_memory_usage()
    runtime_overhead = runtime_end_mem - runtime_start_mem

    peak_memory = tracker.summary()
    print(f"Peak Memory: {peak_memory:.1f} MB")
    print(f"Runtime Memory Overhead: {runtime_overhead:.1f} MB")


if __name__ == "__main__":
    main()



