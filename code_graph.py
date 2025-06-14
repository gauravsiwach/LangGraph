from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

client = OpenAI()

class classifyMessageResponse(BaseModel):
    is_coding_question: bool

class codeAccuracyResponse(BaseModel):
    accuracy_percentage: str

class State(TypedDict):
    user_query: str
    llm_result: str | None
    accuracy_percentage: str | None
    is_coding_question: bool | None

def classify_message(state: State):
    print("🔎classify_message...")
    query = state['user_query']
    SYSTEM_PROMPT = """
    You are a helpful ai assistant, and your job is to detect if user's query is
    related to coding or not.
    return the response in specified JSON boolean only.
    """
    # use pydantic to validate the response
    # sturctured response
    response = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        response_format=classifyMessageResponse
    )
    is_coding_question = response.choices[0].message.parsed.is_coding_question
    state['is_coding_question'] = is_coding_question
    return state

def route_query(state: State) -> Literal["general_query", "coding_query"]:
    print("🔎route_query...")
    is_coding_question = state['is_coding_question']
    
    if is_coding_question:
        return "coding_query"
    else:
        return "general_query"

def general_query(state: State):
    print("🔎general_query...")
    query = state['user_query']
    SYSTEM_PROMPT = """
        You are a helpful AI assistant. reply to the user's query and keep it short and to the point
    """
    # open ai call
    llm_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
    )
    result = llm_response.choices[0].message.content
    state['llm_result'] = result

    return state

def coding_query(state: State):
    print("🔎coding_query...")
    query = state['user_query']
    SYSTEM_PROMPT = """
        You are a coding expert AI assistant. reply to the user's coding query in python and javascript by default.
        if user asks for a specific coding language then reply in that language.
    """
    # open ai call
    llm_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
    )
    result = llm_response.choices[0].message.content
    state['llm_result'] = result

    return state

def coding_validate_query(state: State):
    print("🔎coding_validate_query...")
    query = state['user_query']
    llm_code = state['llm_result']
    SYSTEM_PROMPT = f"""
        You are expert in calculating accuracy of the code according to the user's query.
        reutrn the accuracy percentage of the code in the range of 0 to 100.
        user query: {query}
        code: {llm_code}
    """
    # open ai call
    llm_response = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        response_format=codeAccuracyResponse
    )
    result = llm_response.choices[0].message.parsed.accuracy_percentage
    state['accuracy_percentage'] = result
    return state

graph_builder = StateGraph(State)


# Define the nodes in the graph
graph_builder.add_node("classify_message", classify_message)
graph_builder.add_node("general_query", general_query)
graph_builder.add_node("coding_query", coding_query)
graph_builder.add_node("coding_validate_query", coding_validate_query)
graph_builder.add_node("route_query", route_query)

graph_builder.add_edge(START, "classify_message")
graph_builder.add_conditional_edges("classify_message", route_query)

graph_builder.add_edge("general_query", END)

graph_builder.add_edge("coding_query", "coding_validate_query")
graph_builder.add_edge("coding_query", END)


graph = graph_builder.compile()

def main():
    user = input("Enter your query: ")
    _state = {
        "user_query": user,
        "accuracy_percentage": None,
        "is_coding_question": None,
        "llm_result": None
    }
    graph_result = graph.invoke(_state)
    print("LLM Result:", graph_result)

 
main()