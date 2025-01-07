import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from openai import api_key

from src.agents.prompts import prompt_templates
from src.agents.states.app_state import HultieState, HultieRagState
from src.agents.nodes.app_nodes import create_tool_node_with_fallback, Hultie_planb, init_node, rag_stateinit, \
    retrieve, grade_documents
from src.agents.tools import hultie_tools
from src.agents.transitions.edges import tools_edge_planb
from src.agents.transitions.edges import tools_end_planb

#fecha
load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
memory = MemorySaver()
model_deepseek = ChatOpenAI(model="deepseek-chat", api_key=deepseek_api_key,base_url="https://api.deepseek.com",temperature=0.1, max_tokens=250)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=40)

#Subgrafo encargado de rag
rag_graph = StateGraph(HultieRagState)
rag_graph.add_node("rag_stateinit",rag_stateinit)
rag_graph.add_node("retrieve",retrieve)
rag_graph.add_node("grade_documents",grade_documents)
rag_graph.add_edge(START,"rag_stateinit")
subgraph_rag=rag_graph.compile()

##--------------------------------grafov2-----------------------------------------------------------------------------
planb_tools = [hultie_tools.send_imagen_presentandose,hultie_tools.test_list,hultie_tools.update_sede]
planb_chain = prompt_templates.planb_sysp | model_deepseek



hultie_graph_planb = StateGraph(HultieState)
hultie_graph_planb.add_node("init",init_node)
hultie_graph_planb.add_edge(START,"init")
hultie_graph_planb.add_node("hultie_planb",Hultie_planb(planb_chain,rag_graph=subgraph_rag,verbose=True))
hultie_graph_planb.add_edge("hultie_planb",END)
# hultie_graph_planb.add_node("planb_tools", create_tool_node_with_fallback(planb_tools))
# hultie_graph_planb.add_conditional_edges(
#     "hultie_planb",
#     tools_edge_planb
# )
# hultie_graph_planb.add_conditional_edges(
#     "planb_tools",
#     tools_end_planb,
#
# )

def compilar_planb():
    return hultie_graph_planb.compile(checkpointer=memory)