from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
        model = 'gemini-2.0-flash',
)
prompt = PromptTemplate(
    template = 'Give the answer of the question - {question} according to the this  - {text}',
    input_variables = ['question' ,'text']
)

parser = StrOutputParser()




url = 'https://www.cricbuzz.com/live-cricket-scores/105772/eng-vs-ind-3rd-test-india-tour-of-england-2025'

loader = WebBaseLoader(url)

docs = loader.load()


chain = prompt | model | parser

res = chain.invoke({'question' : 'Who is the POTM ' , 'text' : docs[0].page_content})

print(res)