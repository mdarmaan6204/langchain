from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature = 0.5
)

parser = StrOutputParser()

template = PromptTemplate(
    template = "Write 5 interesting facts about the topic {topic} .",
    input_variables = ['topic']
)

chain = template | model | parser

res = chain.invoke({'topic' : 'Cricket'})
print(res)
chain.get_graph().print_ascii()
