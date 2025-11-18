DOCUMENT_CONFIG = {
    "input_file": "speech.txt",
    "encoding": "utf-8",
    
    "preprocess": {
        "lowercase": False,          
        "remove_special_chars": False, 
        "remove_numbers": False,   
    }
}

CHUNKING_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separators": ["\n\n", "\n", " ", ""],
}

EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",
    "normalize_embeddings": True,
    "batch_size": 32,
}

VECTORSTORE_CONFIG = {
    "persist_directory": "./chroma_db",
    "collection_name": "speech_embeddings",
    "recreate_db": True,
    "chroma_settings": {
        "anonymized_telemetry": False,
    }
}

RETRIEVAL_CONFIG = {
    "k_retrieved_docs": 3,
    "search_type": "similarity",
    "similarity_threshold": 0.0,
    "fetch_k": 20, 
    "lambda_mult": 0.25, 
}

LLM_CONFIG = {
    "model": "mistral",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 512,
    "timeout": 300,
}


PROMPT_TEMPLATES = {
    "standard": """You are a helpful assistant answering questions based on the provided context.

Context:
{context}

Question: {question}

Based only on the context provided above, answer the question. If the answer is not in the context, say "I don't have enough information to answer this question based on the provided text."

Answer:""",

    "summarization": """Based on the provided context, create a concise summary.

Context:
{context}

Provide a 2-3 sentence summary of the key points.

Summary:""",

    "question_generation": """Based on the provided context, generate 5 interesting questions that someone might ask about this content.

Context:
{context}

Generate 5 questions:""",

    "detailed": """You are an expert at explaining complex topics clearly and in detail.

Context:
{context}

Question: {question}

Provide a comprehensive, detailed answer that explains the concepts thoroughly. Reference specific parts of the context where relevant.

Answer:""",
}



LOGGING_CONFIG = {
    "level": "INFO",
    
    "log_file": "rag_system.log",
    
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

SYSTEM_CONFIG = {
    "verbose": True,
    "show_metrics": True,
    "cache_embeddings": True,
    "num_workers": 4,
    
    "seed": 42,
}


RETRIEVAL_STRATEGIES = {
    "query_expansion": {
        "enabled": False,
        "num_expansions": 2,
    },
    

    "hybrid_search": {
        "enabled": False,
        "semantic_weight": 0.7,
        "keyword_weight": 0.3,
    },

    "reranking": {
        "enabled": False,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 3,
    },
}
