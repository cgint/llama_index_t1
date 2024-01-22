from typing import List
from llama_index import Document, StorageContext, VectorStoreIndex, load_index_from_storage
import os


def create_vector(service_context, vector_storage_dir: str, doc_loader: callable) -> List[Document]:
    if not os.path.exists(vector_storage_dir):
        documents = doc_loader()
        print(f"About to build vector-index over {len(documents)} document(s) ...")
        vector_index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context
        )
        print(f"Storing vector-index to {vector_storage_dir} ...")
        vector_index.storage_context.persist(persist_dir=vector_storage_dir)
    else:
        print(f"Loading vector-index from storage from {vector_storage_dir} ...")
        storage_context_vector = StorageContext.from_defaults(persist_dir=vector_storage_dir)
        vector_index = load_index_from_storage(
            service_context=service_context,
            storage_context=storage_context_vector
        )
    return vector_index
