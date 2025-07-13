from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Annotated , Literal , Optional
from pydantic import BaseModel , Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

class Review(BaseModel):
    name : Optional[str] = Field(description =  "Reviewer name")
    sentiment: Literal['pos' , 'neg']  = Field(description = "sentiment of the review either positive , negative or neutral")
    summary: str = Field(description = "A brief summary of review")
    key_frames : Optional[list[str]] = Field(default = None , description = "List of key frames used in review")
    pros : Optional[str] = Field(default = None , description =  "Pros of the review")
    cons : Optional[str] = Field(default = None , description =  "Cons of the review")

structured_model = model.with_structured_output(Review)

res = structured_model.invoke("""Overpriced and underwhelming. The camera improvements are minimal compared to iPhone 15. Experienced several app crashes in the first week. The phone randomly restarts sometimes. Customer service was unhelpful when I reported these issues. Considering returning it and switching to Android. Reviw by armaan""")

print(res)