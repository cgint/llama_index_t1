import os
def get_llm(llm_engine, llm_model, openai_model):
    temperature = 0.1
    if llm_engine == "together":
        from llama_index.llms import OpenAILike
        print(f"About to instanciate LLM {openai_model} using Together.ai ...")
        return OpenAILike(
            model=openai_model,
            api_base="https://api.together.xyz",
            api_key=os.getenv("TOGETHER_AI_KEY"),
            is_chat_model=True,
            is_function_calling_model=True,
            reuse_client=False, # When doing anything with large volumes of async API calls, setting this to false can improve stability.",
            max_retries=10,
            timeout=120,
            temperature=temperature
        )
    elif llm_engine == "openai":
        from llama_index.llms import OpenAI
        print(f"About to instanciate LLM {openai_model} using OpenAI ...")
        return OpenAI(
            model=openai_model,
            #api_base=api_base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=temperature
        )
    elif llm_engine == "ollama":
        from llama_index.llms import Ollama
        api_base_url = "http://192.168.0.99:11434"
        print(f"About to instanciate LLM {llm_model} on {api_base_url} using Ollama ...")
        return Ollama(
            model=llm_model, 
            base_url=api_base_url, 
            request_timeout=900, 
            temperature=temperature,
            #additional_kwargs={"main_gpu": 1} # see https://github.com/jmorganca/ollama/issues/1813#issuecomment-1902682612
        )
    elif llm_engine == "ollama-gpu0":
        # Needs an Ollama-instance starting with this command: "CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11535 ollama serve"
        from llama_index.llms import Ollama
        api_base_url = "http://192.168.0.99:11430"
        print(f"About to instanciate LLM {llm_model} on {api_base_url} using Ollama-Instance with GPU-ID 0 ...")
        return Ollama(model=llm_model, base_url=api_base_url, request_timeout=900, temperature=temperature)
    elif llm_engine == "ollama-gpu1":
        # Needs an Ollama-instance starting with this command: "CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11535 ollama serve"
        from llama_index.llms import Ollama
        api_base_url = "http://192.168.0.99:11431"
        print(f"About to instanciate LLM {llm_model} on {api_base_url} using Ollama-Instance with GPU-ID 1 ...")
        return Ollama(model=llm_model, base_url=api_base_url, request_timeout=900, temperature=temperature)
    else:
        raise Exception(f"Unchronos-hermes-13bknown llm_engine: {llm_engine}")