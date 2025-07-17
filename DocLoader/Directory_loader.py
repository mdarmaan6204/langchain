from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

loader = DirectoryLoader(
    path = 'D:\Sem 8',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.lazy_load()




for documents in docs:
    print(documents.metadata)