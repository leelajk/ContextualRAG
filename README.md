# AmbedkarGPT-Intern-Task

Simple command-line Q&A system that ingests a short Dr. B.R. Ambedkar speech and answers questions **only** using that text (RAG pipeline).

## Files
- `main.py` : CLI application (Python).
- `speech.txt` : Provided speech excerpt (already included).
- `requirements.txt` : Python dependencies.

## Requirements
- Python: 3.8+ (Recommended 3.11)
- RAM:
    4GB+ for lightweight models (Orca-Mini)
    8GB+ for Mistral 7B
- Disk Space: ~10GB for large Ollama models
- Ollama Installed: https://ollama.com
- Input File: Plain text (.txt)

## Usage
- 'Interactive Mode'
- Load a text file → ask any question → get contextual answers.

## Batch Mode
- Automate multiple queries for reports or research workflows (optional to add in future versions).
