import logging
import sys
import os

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)

# context, mode, question, answer (mode could be "query" or "chat")
be_verbose = True

# define LLM
embed_model_name = os.getenv("CONF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2") # "BAAI/bge-small-en-v1.5"
llm_engine = os.getenv("CONF_TYPE", "ollama") # "together" # "openai" # 
llm_models = os.getenv("CONF_MODEL", "codeup").split(",") # ["zephyr:7b-beta-fp16"] # ["llama-pro:8b-instruct-fp16", "mistral-openorca:7b-fp16", "tinyllama", "openchat"] # ["mistral-openorca:7b-fp16", "notux:8x7b-v1-q3_K_M"] # ["dolphin-mixtral:8x7b-v2.7-q3_K_M", "mistral-openorca"] # "dolphin-mistral:7b-v2.6-dpo-laser-fp16" # "mixtral" # "gpt-4-1106-preview" # "mixtral:8x7b-instruct-v0.1-q3_K_M" # "neural-chat" # "mixtral-together" # "stablelm-zephyr" # "solar" # "mixtral:8x7b-instruct-v0.1-q4_K_M"  # "gpt-3.5-turbo-1106" # "gpt-4-1106-preview" # 
openai_model = os.getenv("CONF_AI_MODEL", "ignore" ) # "mistralai/Mistral-7B-Instruct-v0.2" # "togethercomputer/llama-2-13b-chat" # "mistralai/Mixtral-8x7B-v0.1"
run_scenario = os.getenv("CONF_IDENT", "AY-yahoo-content-no_sentiment-40") # "chap-per-meldung"
### Good ones: openchat, codeup, dolphin-mistrals

logging.info(f"Running with\n embed_model_name={embed_model_name}\n llm_engine={llm_engine}\n llm_models={llm_models}\n openai_model={openai_model}\n run_scenario={run_scenario}")

logging.info("Initialzing RAG ...")
from lib.rag_process import run_for_config, init_rag_process
init_rag_process()
logging.info("Initialzing RAG done. Starting loop ...")
for llm_model in llm_models:
    logging.info(f"\nRunning for {llm_model} ...")
    run_identifier = f"{llm_model}_{run_scenario}"
    try:
        run_for_config(embed_model_name, llm_engine, llm_model, openai_model, run_identifier, be_verbose)
    except Exception as e:
        logging.error(f"Error for {run_identifier}: {e}")
        # save to error-file
        with open(f"/data/error-{run_identifier}.txt", "a") as f:
            f.write(f"Error for {run_identifier}: {e}\n")
