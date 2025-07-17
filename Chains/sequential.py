from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash',
)

parser = StrOutputParser()

promt1 = PromptTemplate(
    template = 'Generate a detailed paragrapgh about {topic}.',
    input_variables = ['topic']
)

promt2 = PromptTemplate(
    template = 'Generate the top 3 importants lines from the text below \n {text}',
    input_variables = ['text']
)

promt3 = PromptTemplate(
    template = 'Gneerate top 5 key words used in these lines {text}',
    input_variables = ['text']
)

chain = promt1 | model | parser | promt2 | model | parser | promt3 | model | parser

res = chain.invoke({'topic' : 'Cricket'})

print(res)

chain.get_graph().print_ascii()