from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    query: str
    llm_result: str


def chat_bot(state: State):
    # open ai call
    result= "Hello, this is a response from the chat bot."
    state['llm_result'] = result

    return state

graph_builder = StateGraph(State)

graph_builder.add_node("chat_bot", chat_bot)

graph_builder.add_edge(START, "chat_bot")
graph_builder.add_edge("chat_bot", END)

graph= graph_builder.compile()

def main():
    user = input("Enter your query: ")
    _state = {
        "query": user,
        "llm_result": ""
    }
    graph_result = graph.invoke(_state)
    print("LLM Result:", graph_result['llm_result'])

 
main()