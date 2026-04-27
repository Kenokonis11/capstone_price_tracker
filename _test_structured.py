from dotenv import load_dotenv
load_dotenv(override=True)
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class Test(BaseModel):
    name: str = Field(...)

m = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
)
s = m.with_structured_output(Test)
print("Calling structured output...")
r = s.invoke("Name a fruit")
print("Got:", r)
