from langgraph.graph import END,START,StateGraph
from langgraph.prebuilt import ToolNode,tools_condition

from recon.state import AgentState
from recon.models import ReconOut
from recon.agents import fetcher
from recon.tools import fetch_feed

from langchain_core.messages import HumanMessage

tools = [fetch_feed]
graph = StateGraph(AgentState)

graph.add_node("fetcher",fetcher)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START,"fetcher")
graph.add_conditional_edges("fetcher",tools_condition)
graph.add_edge("tools", "fetcher")

app=graph.compile()

if __name__ =="__main__":
    initial_state = {
        "messages" : [HumanMessage("find about IFS company")]
    }
    result = app.invoke(initial_state)

    print(result["messages"][-1].content)