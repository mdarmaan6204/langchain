from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
        model = 'gemini-2.0-flash',
)
loader = TextLoader('./DocLoader/cricket.txt' , encoding = 'utf-8')

docs = loader.load()

prompt = PromptTemplate(
    template = 'Suggest single title for the poem - {poem}',
    input_variables = ['peom']
)

parser = StrOutputParser()

chain = prompt | model | parser


print(docs)

res = chain.invoke({'poem' : docs[0].page_content})

print(res)
