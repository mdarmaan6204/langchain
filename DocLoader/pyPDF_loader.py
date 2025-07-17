from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("D:\Sem 8\A Comparative Study of MobileNetV2 and Xception for Deepfake Detection Using Transfer Learning2.pdf")

docs = loader.load()

print(len(docs))
print(docs[5].page_content)