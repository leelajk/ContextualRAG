"""
RAG System with LangChain, ChromaDB, and Ollama
This script implements a Retrieval-Augmented Generation pipeline that:
1. Loads text from speech.txt
2. Splits the text into manageable chunks
3. Creates embeddings using HuggingFace transformers
4. Stores embeddings in ChromaDB vector database
5. Retrieves relevant chunks based on user queries
6. Generates answers using Ollama with Mistral 7B
"""

import os
import sys
from typing import List, Dict
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGSystem:
    
    def __init__(self, 
                 speech_file: str = "speech.txt",
                 persist_directory: str = "./chroma_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 k_retrieved_docs: int = 3):
        self.speech_file = speech_file
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieved_docs = k_retrieved_docs
        
        self.documents = None
        self.chunks = None
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.rag_chain = None
        
        print("RAG System initialized with configuration:")
        print(f"  - Speech file: {speech_file}")
        print(f"  - Persist directory: {persist_directory}")
        print(f"  - Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        print(f"  - Retrieved documents: {k_retrieved_docs}\n")
    
    def load_documents(self) -> List:
        print(f"[STEP 1] Loading documents from '{self.speech_file}'...")
        
        if not os.path.exists(self.speech_file):
            raise FileNotFoundError(
                f"Speech file not found: {self.speech_file}\n"
                f"Please ensure 'speech.txt' exists in the current directory."
            )
        
        loader = TextLoader(self.speech_file, encoding='utf-8')
        self.documents = loader.load()
        
        print(f"✓ Loaded {len(self.documents)} document(s)")
        print(f"  Total content length: {sum(len(doc.page_content) for doc in self.documents)} characters\n")
        
        return self.documents
    
    def split_documents(self) -> List:
        print(f"[STEP 2] Splitting documents into chunks...")
        print(f"  Configuration: size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.chunks = splitter.split_documents(self.documents)
        
        print(f"✓ Created {len(self.chunks)} chunks")
        print(f"  Average chunk size: {sum(len(chunk.page_content) for chunk in self.chunks) / len(self.chunks):.0f} characters\n")
        
        return self.chunks
    
    def create_embeddings(self) -> HuggingFaceEmbeddings:
        print("[STEP 3] Creating embeddings with HuggingFace...")
        print("  Model: sentence-transformers/all-MiniLM-L6-v2")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("✓ Embeddings model loaded successfully\n")
        
        return self.embeddings
    
    def create_vector_store(self) -> Chroma:
        print("[STEP 4] Creating ChromaDB vector store...")
        print(f"  Persist directory: {self.persist_directory}")
        
        # Remove existing database if it exists to ensure fresh start
        if os.path.exists(self.persist_directory):
            print("  Note: Existing database will be overwritten")
        
        self.vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"✓ Vector store created with {len(self.chunks)} embeddings")
        print(f"✓ Database persisted to '{self.persist_directory}'\n")
        
        return self.vector_store
    
    def initialize_llm(self) -> Ollama:
        print("[STEP 5] Initializing Ollama LLM (Mistral 7B)...")
        print("  Note: Ensure Ollama is running on localhost:11434")
        print("  If not running, start it with: ollama serve")
        
        self.llm = Ollama(
            model="orca-mini",
            base_url="http://localhost:11434",
            temperature=0.7,
            top_p=0.9,
            top_k=40
)

        
        print("✓ Ollama LLM initialized\n")
        
        return self.llm
    
    def build_rag_chain(self) -> None:
        print("[STEP 6] Building RAG chain...\n")
        
        # Create retriever from vector store
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k_retrieved_docs}
        )
        template = """You are a helpful assistant answering questions based on the provided context.

Context:
{context}

Question: {question}

Based only on the context provided above, answer the question. If the answer is not in the context, say "I don't have enough information to answer this question based on the provided text."

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Build the chain using LCEL (LangChain Expression Language)
        # This chains together: retrieval -> formatting -> LLM -> output parsing
        self.rag_chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ RAG chain built successfully")
        print(f"✓ System ready for queries\n")
    
    @staticmethod
    def _format_docs(docs: List) -> str:
        return "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
    
    def query(self, question: str) -> str:
        if self.rag_chain is None:
            raise RuntimeError("RAG chain not built. Call build_rag_chain() first.")
        
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        answer = self.rag_chain.invoke(question)
        
        print(f"Answer:\n{answer}")
        print(f"{'='*80}\n")
        
        return answer
    
    def setup(self) -> None:
        print("\n" + "="*80)
        print("INITIALIZING RAG SYSTEM".center(80))
        print("="*80 + "\n")
        
        self.load_documents()
        self.split_documents()
        self.create_embeddings()
        self.create_vector_store()
        self.initialize_llm()
        self.build_rag_chain()
        
        print("="*80)
        print("SETUP COMPLETE - SYSTEM READY".center(80))
        print("="*80 + "\n")


def interactive_query_loop(rag_system: RAGSystem) -> None:
    print("Interactive Query Mode")
    print("-" * 80)
    print("Type your questions below. Type 'exit' or 'quit' to end.\n")
    
    while True:
        try:
            user_input = input("Your question: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nExiting RAG system. Goodbye!")
                break
            
            if not user_input:
                print("Please enter a valid question.\n")
                continue
            
            rag_system.query(user_input)
            
        except KeyboardInterrupt:
            print("\n\nExiting RAG system. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")
            print("Please try again.\n")


def main():
    
    # Initialize RAG system
    rag = RAGSystem(
        speech_file="speech.txt",
        persist_directory="./chroma_db",
        chunk_size=1000,
        chunk_overlap=200,
        k_retrieved_docs=3
    )
    
    try:
        # Setup the system
        rag.setup()
        
        example_questions = [
            # "What is the main topic of the speech?",
            # "Who is the speaker?",
            # "What are the key points discussed?",
        ]
        
        for question in example_questions:
            rag.query(question)
        
        # Start interactive query loop
        interactive_query_loop(rag)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\nPlease ensure Ollama is running:")
        print("  1. Install Ollama from https://ollama.ai")
        print("  2. Run: ollama serve")
        print("  3. In another terminal, pull Mistral: ollama pull mistral")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
