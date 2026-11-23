from recon.state import AgentState 
from recon.config import get_llm
from recon.tools import fetch_feed

def fetcher(state: AgentState):
    """Fetcher agent for the project Recon
    this agents responisbility is to get the news articles using the feedparser 
    tool and create the Ariticle Object and pass to the next agent
    """
    
    llm_with_tools = get_llm("llama_cpp").bind_tools([fetch_feed])
    
    response= llm_with_tools.invoke(state.messages)