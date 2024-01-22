from datetime import datetime

def process_queries_with_query_engine(engine, context, queries):
  return _process_queries_with_engine(engine, "query", context, queries)

def process_queries_with_chat_engine(engine, context, queries):
  return _process_queries_with_engine(engine, "chat", context, queries)
  

def _engine_query_print(query_engine, question):
    answer = query_engine.query(question)
    print("\n------------------------------------")
    print(f"\nInput: \033[1m{question}\033[0m")
    print(f"\nOutput: \033[1m{answer}\033[0m")
    print("------------------------------------\n")
    return answer

def _engine_chat_print(chat_engine, question):
    answer = chat_engine.chat(question)
    print("\n------------------------------------")
    print(f"\nInput: \033[1m{question}\033[0m")
    print(f"\nOutput: \033[1m{answer}\033[0m")
    print("------------------------------------\n")
    return answer

def _engine_to_context_question_answer(query_engine, type, context, question):
  if type == "query":
    answer = _engine_query_print(query_engine, question)
  elif type == "chat":
    answer = _engine_chat_print(query_engine, question)
  else:
    raise Exception(f"Unknown type: {type}. Must be 'query' or 'chat'")
  return (context, type, question, answer, datetime.now())

def engine_chat_to_context_question_answer(query_engine, context, question):
  answer = _engine_chat_print(query_engine, question)
  return (context, "chat", question, answer, datetime.now())

def _process_queries_with_engine(engine, type, context, queries):
  data = []
  for query in queries:
    try:
      data.append(_engine_to_context_question_answer(engine, type, context, query))
    except Exception as e:
     data.append( (context, type, query, str(e), datetime.now()))
  return data
