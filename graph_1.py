from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

class State(TypedDict):
    query: str
    llm_result: str


def chat_bot(state: State):
    query = state['query']
    # open ai call
    llm_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
    )

    result = llm_response.choices[0].message.content.strip()
    state['llm_result'] = result

    return state

graph_builder = StateGraph(State)

graph_builder.add_node("chat_bot", chat_bot)

graph_builder.add_edge(START, "chat_bot")
graph_builder.add_edge("chat_bot", END)

graph = graph_builder.compile()

def main():
    user = input("Enter your query: ")
    _state = {
        "query": user,
        "llm_result": ""
    }
    graph_result = graph.invoke(_state)
    print("LLM Result:", graph_result['llm_result'])

 
main()