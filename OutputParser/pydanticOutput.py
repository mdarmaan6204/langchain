from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel , Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name : str = Field(description = 'Name of a person')
    age : int = Field(gt = 20 , description = 'Age of the person')
    city : str = Field(description = 'Location city of the person belongs to')

parser = PydanticOutputParser(pydantic_object = Person)

template = PromptTemplate(
    template = '''Generate a person from {country}. 
    Return ONLY valid JSON format. No code, no explanations, no markdown.
    
    {format_instruction}''',
    input_variables = ['country'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

try:
    res = chain.invoke({'country' : 'India'})
    print(res)
    print(f"Name: {res.name}")
    print(f"Age: {res.age}")
    print(f"City: {res.city}")
except Exception as e:
    print(f"Error: {e}")
    # See raw output without parser
    chain_no_parser = template | model
    raw_output = chain_no_parser.invoke({'country' : 'India'})
    print("Raw output:", raw_output)