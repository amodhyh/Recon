from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from recon.models import Article
from recon.state import AgentState


def test_state_defaults_are_empty():
    state = AgentState()
    assert state.history == []
    assert state.competitors == []
    assert state.articles == []
    assert state.retrieved_docs == []
    assert state.brief == ""
    assert state.next_agent == ""


def test_state_defaults_are_not_shared_between_instances():
    a, b = AgentState(), AgentState()
    a.competitors.append("Acme")
    assert b.competitors == []


def test_state_accepts_articles():
    article = Article(competitor="Acme", title="Launch", url="https://acme.test")
    state = AgentState(articles=[article])
    assert state.articles[0].competitor == "Acme"


def test_history_appends_through_graph():
    def respond(state: AgentState) -> dict:
        return {"history": [AIMessage(content="reply")]}

    graph = StateGraph(AgentState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)

    result = graph.compile().invoke({"history": [HumanMessage(content="hi")]})

    assert [m.content for m in result["history"]] == ["hi", "reply"]
