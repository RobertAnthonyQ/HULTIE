from asyncio import AbstractEventLoop
from typing import List, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict



class HultieState(TypedDict):
    #Información del usuario
    nombre: str # Nombre del usuario->Default: Nombre que aparece en su wsp
    cel: str
    question: str # Última pregunta realizada por el usuario
    inscrito: bool #Bool si está inscrito o no el usuario

    #Información de la sede
    sede: str # Sede del usuario
    campus_director: str  # Campus Director de la sede del usuario
    whatsapp: str  # link de grupo de wsp
    fecha_limite_postulacion: str  # Fecha máxima de postulación relacionada con la sede del usuario
    link_postulacion : str
    instagram: str
    tiktok: str

    #Información de la respuesta del agente
    generation: str # Última respusta generada por hultie

    #Utils
    documents: List[str] # Lista de documentos relevantes extraídos del vectorstore
    tries_retrieving: int # Número de intentos realizados para recuperar info relevanta
    messages: Annotated[list[AnyMessage], add_messages] # Lista de mensajes que forman parte del contexto de la conversación.

class HultieRagState(TypedDict):
    question: str  # Última pregunta realizada por el usuario
    generation: str  # Última respusta generada por hultie
    sede: str
    documents: List[str]  # Lista de documentos relevantes extraídos del vectorstore
    tries_retrieving: int # Número de intentos realizados para recuperar info relevanta
    messages: Annotated[list[AnyMessage], add_messages]







