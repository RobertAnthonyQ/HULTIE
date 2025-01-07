import asyncio
import os
import sys
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
import random

from langgraph.constants import END
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.agents.states.app_state import HultieState
from utils.general_utils import print_colored
from utils.notion_utils import obtener_enlaces_disponibles, obtener_datos_sede
from src import EvolutionAPI, SEDES_CONFIG
from src.agents.utils.agent_utils import notool_update_sede

verbose_tools = True #Poner en true si es que se quieren activar los logs

sedes_disponible = ["CERTUS", "UCST", "UPC", "UPCH", "UNCP", "UNI", "UP", "USIL", "UDEP", "UCSUR", "UARM", "ULIMA", "ESAN", "UCAL", "UTEC", "PUCP", "UNMSM", "UNAP", "UCSM", "UTP"]

#-------------------------state0---------------------------------------
#core
def check_inscription(name : str):
    "Llamar a esta función cuando el usuario diga su nombre para verificar si ya se ha inscrito o no"
    #inscrito = random.choice([True, False])
    inscrito = False
    #SIMULACIÓN:
    if inscrito:
        sede = "PUCP"
        body_urls = obtener_datos_sede(sede)
        link_postulacion = body_urls["link_postulacion"]
        campus_director = body_urls["campus_director"]
        inscripciones_abiertas = body_urls["etiquetas"]
        fecha_limite_postulacion= body_urls["fecha_limite_postulacion"]
        url_bases = body_urls["bases_url"]
        url_instagram= body_urls["instagram"]
        url_tiktok = body_urls["tiktok"]
        url_whatsapp = body_urls["whatsapp"]
    else:
        sede = ""
        link_postulacion = ""
        campus_director = ""
        inscripciones_abiertas = ""
        fecha_limite_postulacion= ""
        url_bases = ""
        url_instagram= ""
        url_tiktok = ""
        url_whatsapp = ""
    return {"inscrito": inscrito,"sede" : sede}


#-------------------------state2---------------------------------------
@tool
def update_sede(state: Annotated[dict,InjectedState],sede : str):
    """
    Hace un fetch al database para conseguir los urls importantes y las bases de la sede del usuario, por ahora se tiene: Las bases, el link de postulación, el instagram y el tiktok de hult prize de la sede correspondiente
    Args: Sede actual del usuario
    """
    tool_calls = state["messages"][-1].additional_kwargs["tool_calls"]
    tool_call_id = state["messages"][-1].additional_kwargs["tool_calls"][-1]["id"] #Por seacaso, ya que en la mayoria de veces solo se llama una vez
    for tool_call in tool_calls: #Comprobamos que el tool_id corresponda a esta función
        if tool_call["function"]["name"] == "update_sede":
            print_colored("Corrigiendo tool_call id",31)
            tool_call_id = tool_call_id

    if verbose_tools: print_colored(f"---Fetching Links for {sede}----",33)
    campus_director= ""
    url_whatsapp = ""
    url_instagram = ""
    url_tiktok = ""
    fecha_limite_postulacion = ""
    link_postulacion = ""
    if sede not in sedes_disponible:
        sedes_str = ", ".join(sedes_disponible)
        #aca poner un agente con salida estructurada
        if verbose_tools: print_colored(f"Sede no disponible,se paso la sede {sede} y las sedes disponibles son {sedes_str}",31)
        return f"La sede que menciono el usuario no esta dentro de la lista de sedes disponibles. Recordarle al usuario las sedes disponibles: {sedes_str}"
    try:
        body_urls = obtener_datos_sede(sede)
        link_postulacion = body_urls["link_postulacion"]
        campus_director = body_urls["campus_director"]
        inscripciones_abiertas = body_urls["etiquetas"]
        fecha_limite_postulacion= body_urls["fecha_limite_postulacion"]
        url_bases = body_urls["bases_url"]
        url_instagram= body_urls["instagram"]
        url_tiktok = body_urls["tiktok"]
        url_whatsapp = body_urls["whatsapp"]
        inscrito = state["inscrito"]

        urls = f"-----Información del onCampus Program de {sede}----\n Campus Director(líder) del programa de {sede}: {campus_director}\nEstado de postulación de {sede}:{inscripciones_abiertas}\nFecha Límite de postulación de {sede}:{fecha_limite_postulacion}\nLink para descargar las bases de {sede}: {url_bases}\nLink de postulación de {sede}: {link_postulacion}\n Link de tiktok de {sede}: {url_tiktok}\n Link de instagram de {sede}: {url_instagram}"
    except:
        print_colored("Error al hacer el fetch de info de sede",31)


    return Command(
        update={
            "messages": [ToolMessage(
                f"Sede del usuario actualizada. Información de la sede(aparte de la que se encuentra en las bases):\n{urls}. URGENTE: Enviarle toda la información de los urls al usuario, pero en el mismo mensaje seguir el flujo que estipula en el system prompt.",
                tool_call_id=tool_call_id)],
            "sede": sede,
            "campus_director": campus_director,
            "instagram": url_instagram,
            "tiktok": url_tiktok,
            "fecha_limite_postulacion": fecha_limite_postulacion,
            "whatsapp": url_whatsapp
        }
    )






#-------------------------------------evolution-api-tools-------------------------
api = EvolutionAPI()
TEST_PHONE_NUMBER="51923559154"


def test_list(cel):
    """Envio de lista con las sedes disponibles"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = asyncio.get_event_loop()
    print_colored("Se esta enviando la lista...", 31)
    loop.run_until_complete(api.send_list(number=cel, universities=SEDES_CONFIG))
    loop.stop()
    loop.close()
    print_colored("Se envio la lista", 33)




@tool
def send_imagen_presentandose(state: Annotated[dict,InjectedState],caption : str):
    """Envio de imagen para presentarse.
    Args:
        caption: str -> Caption de la imagen, aca deberás presentarte.
    """

    tool_calls = state["messages"][-1].additional_kwargs["tool_calls"]
    print_colored(f"tool_calls:{tool_calls}",33)
    tool_call_id = state["messages"][-1].additional_kwargs["tool_calls"][-1]["id"] #Por seacaso, ya que en la mayoria de veces solo se llama una vez
    for tool_call in tool_calls: #Comprobamos que el tool_id corresponda a esta función
        if tool_call["function"]["name"] == "send_imagen_presentandose":
            print_colored("Corrigiendo tool_call id",31)
            tool_call_id = tool_call_id


    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = asyncio.get_event_loop()
    print_colored(f"Enviando imagenes: {caption}", 31)
    loop.run_until_complete(api.send_image(number="51923559154",image_url="https://i.pinimg.com/originals/fc/b3/86/fcb3866603194921e56669d0620f42d5.png",caption=caption))
    loop.stop()
    loop.close()


    return Command(
                update={
                    "messages": [ToolMessage(
                f"Imagen y presentación enviada/graph_ended",
                tool_call_id=tool_call_id)]
                },
                goto=END
            )




