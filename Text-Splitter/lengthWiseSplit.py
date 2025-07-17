from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    'D:\Sem 8\A Comparative Study of MobileNetV2 and Xception for Deepfake Detection Using Transfer Learning2.pdf'
)

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 200 , 
    chunk_overlap = 5,
    separator = ' '
)

res = splitter.split_documents(docs)

print(res[1])