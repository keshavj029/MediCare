from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
import logging
import re
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use a simple OpenAI model for testing
groq_api_key = os.getenv("Groq_api_key")

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name='mixtral-8x7b-32768'
)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

class SymptomData(BaseModel):
    symptoms: str

symptoms_doctor = {'Eye': ["Robin Kahn", 8762037849],
                   'Cold and Headache':["Sam keatings", 986394809],
                   }


@app.post("/consult")
async def consult(data: SymptomData):
    health_template = PromptTemplate(
        input_variables=['chat_history', 'symptoms'],
        template="""
        You are a helpful and friendly health chatbot. Based on the user's input:

Input: {symptoms}

1. If the input is a casual greeting like "Hey," "Hi," "Hello," or similar, respond warmly and ask how they are doing.

2. If the input is irrelevant or outside the scope of health-related queries, politely inform the user that you are unable to answer those questions.

3. If the input describes health symptoms, provide a thorough analysis. Mention if the symptoms could indicate any serious health issues that may require consulting a doctor. Suggest possible home remedies if applicable.

4. If the symptoms are severe, recommend consulting a doctor. For example:
   - For eye problems: Robin Kahn, contact - 8646530986
   - For cold and headache: Sam Keatings, contact - 0098643209

Respond directly to the user with only the content of your response.
        """
    )

    try:
        health_chain = LLMChain(llm=llm, prompt=health_template, verbose=True, output_key='output', memory=memory)
    except Exception as e:
        logging.error(f"Error creating LLMChain: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating LLMChain: {e}")

    try:
        response = health_chain({
            'symptoms': data.symptoms
        })
    except Exception as e:
        logging.error(f"Error during LLMChain execution: {e}")
        raise HTTPException(status_code=500, detail=f"Error during LLMChain execution: {e}")
    health_advice = response['output']

    # Remove double newlines
    health_advice_cleaned = re.sub(r'\n\n', ' ', health_advice)

    # Optionally, you can also replace single newlines with spaces
    health_advice_cleaned = health_advice_cleaned.replace('\n', ' ')
    health_advice_cleaned = health_advice_cleaned.replace('*', ' ')

    # If you want to ensure there are no multiple consecutive spaces
    health_advice_cleaned = re.sub(r'\s+', ' ', health_advice_cleaned)

    return {"health_advice": health_advice_cleaned}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("your_script_name:app", host="0.0.0.0", port=8000, reload=True)
