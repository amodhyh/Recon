from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    agent_name: str = ""
    message: str = ""
    history: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


def add_message(state: AgentState) -> dict:
    """Graph node: append the current `message` to `history`.

    `history` uses the `add_messages` reducer, so returning a partial update
    appends (and de-duplicates by id) rather than overwriting.
    """
    if not state.message:
        return {}
    return {"history": [HumanMessage(content=state.message, name=state.agent_name or None)]}
