import time
from typing import List
from llama_index import Document, KnowledgeGraphIndex, StorageContext, load_index_from_storage
import os
from pyvis.network import Network
from llama_index.graph_stores import SimpleGraphStore

def create_graph(service_context, graph_storage_dir: str, graph_output_file: str, doc_loader: callable) -> List[Document]:
    if not os.path.exists(graph_storage_dir):
        documents = doc_loader()
        print(f"About to build graph-index over {len(documents)} document(s) ...")
        build_start = time.time()        
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            service_context=service_context,
            max_triplets_per_chunk=40,
            max_object_length=4096,
            show_progress=True,
            include_embeddings=True
        )
        build_end = time.time()
        print(f"Building graph-index took {build_end-build_start} seconds.")
        kg_index.storage_context.persist(persist_dir=graph_storage_dir)

        if graph_output_file is not None:
            print(f"Storing graph-index to {graph_storage_dir} ...")
            net = Network(height="900px", notebook=False, directed=True)
            net.from_nx(kg_index.get_networkx_graph())
            net.save_graph(graph_output_file)
    else:
        print(f"Loading graph-index from storage from {graph_storage_dir} ...")
        storage_context = StorageContext.from_defaults(persist_dir=graph_storage_dir)
        kg_index = load_index_from_storage(
            storage_context=storage_context,
            service_context=service_context
        )
    return kg_index
