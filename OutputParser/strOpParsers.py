from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm = llm)

tmp1 = PromptTemplate(
    template = 'Write a detailed analyis on the topic of {topic}',
    input_varaibales = ['topic']
)

tmp2 = PromptTemplate(
    template = 'Write 3 lines usmmary on the following text \n {text}',
    input_varaibles = ['text']
)

parser = StrOutputParser()

chain = tmp1 | model | parser | tmp2 | model | parser 

res = chain.invoke({'topic' : 'black hole'})
print(res)