from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from  src.tools.tool import scrape_url,web_search
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["HF_TOKEN"] = os.getenv("HF_API_KEY")

# 2. Change the task from "text-generation" to "conversational"
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",  # <-- Changed here to fix the value error
    
    temperature=0
)


# 3. FIX: Change 'llm_wrapper' to 'llm'
llm = ChatHuggingFace(llm=llm_endpoint)






# 1st Agent : Search Agent
def build_search_agent():
    return create_agent(
        model= llm,
        tools=[web_search],
       
    )



# 2nd Agent : Reader Agent
def build_reader_agent():
    return create_agent(
        model= llm,
        tools=[scrape_url],

    )



#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()





#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()