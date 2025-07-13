from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = '''Create a fictional person and return ONLY a JSON object {format_instruction}''',
    input_variables = [] ,
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

res = chain.invoke({})
print(res)
print(res['name'])