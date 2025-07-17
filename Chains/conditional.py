from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from pydantic import BaseModel , Field
from typing import Literal
load_dotenv()


model = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash',
)

class FeedBack(BaseModel):
    sentiment : Literal['positive' , 'negative'] = Field(description = 'Give a sentiment of text')

    

parser1 = PydanticOutputParser(pydantic_object = FeedBack)

prompt1 = PromptTemplate(
    template = 'Give the sentiment of the feedback {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction' : parser1.get_format_instructions()}
)

parser2 = StrOutputParser()

classifier_chain = prompt1 | model | parser1

prompt2 = PromptTemplate(
    template = 'Write a proper response for this positive feedback {feedback} only in single string no extra explanation',
    input_variables = ['feedback']   
)

prompt3 = PromptTemplate(
    template = 'Write a proper response for this negative feedback {feedback} only in single string no extra explanation',
    input_variables = ['feedback']   
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive' , prompt2 | model | parser2),
    (lambda x: x.sentiment == 'negative' , prompt3 | model | parser2),
    RunnableLambda(lambda x : "Could find sentiment")
)

# print(classifier_chain.invoke({'feedback' : "This is a very good house."}).sentiment)

final_chain = classifier_chain | branch_chain

res = final_chain.invoke({'feedback' : "This is a good house."})
print(res)

final_chain.get_graph().print_ascii()


