# llama_index_t1
First steps with llama index to use as graph/vector semantic search with reranking, sub-queries, ...

# Main code
Files are in `llamaindex_simple_graph_rag.py` and `lib/*`.

This Python script is a driver for a system that processes data using language models. It uses environment variables for configuration, initializes a Retrieval-Augmented Generation (RAG) system, and runs a process for each specified language model.

The process includes keyword and vector tools for answering questions about relationships and semantic similarity, respectively. It also uses various selectors and a response synthesizer for data handling. 

Errors are logged and written to an error file. The script is used for running various language models in a specific scenario to analyze text data.

## How to run
```
Usage: ./build_run.sh <type> <model> <ai_model> <ident>
  Examples:
       ./build_run.sh together mixtral-together mistralai/Mixtral-8x7B-Instruct-v0.1 AY-yahoo-content-no_sentiment-40
       ./build_run.sh ollama codeup ignore AY-yahoo-content-no_sentiment-40
```

# File 'ast_test.py'
Status: Draft

This script is designed to analyze and report the method calls made within each function across multiple Python files.
Its primary purpose is to provide an overview of how different functions interact with other parts of the code,
specifically focusing on which methods are called within each function. 

This analysis can be useful for understanding code structure, debugging, or for refactoring purposes.
Essentially, it creates a map of method usage throughout the given Python files.