from langchain_core.messages import ToolMessage

from utils.general_utils import print_colored
from utils.notion_utils import obtener_datos_sede

verbose_utils = True
def handle_tool_error(state: dict) -> dict:
    """
    Maneja errores de herramientas en el flujo de trabajo del grafo.

    :param state: El state debe contener por lo menos:
                  - "messages": Una lista de mensajes, donde el último mensaje será la respuesta de la toolcall, el cual será
                  el error.
    :return: En vez de parar la ejecución,  empaquetamos este error en un ToolMessage para mandarselo al agente.
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def notool_update_sede(sede : str):
    """
    Hace un fetch al database para conseguir los urls importantes y las bases de la sede del usuario, por ahora se tiene: Las bases, el link de postulación, el instagram y el tiktok de hult prize de la sede correspondiente
    Args: Sede actual del usuario
    """

    if verbose_utils: print_colored(f"---Fetching Links for {sede}----",33)
    campus_director= ""
    url_whatsapp = ""
    url_instagram = ""
    url_tiktok = ""
    url_bases = ""
    fecha_limite_postulacion = ""
    link_postulacion = ""

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
        urls = f"Sede: {sede}\n Campus Director(cabeza) del programa: {campus_director}\nestado:{inscripciones_abiertas}\nFecha Límite de postulación:{fecha_limite_postulacion}\nLink para descargar las bases: {url_bases}\nLink de postulación: {link_postulacion}\n Link de tiktok: {url_tiktok}\n Link de instagram: {url_instagram}"
    except:
        print_colored("Error al hacer el fetch de info de sede",31)

    return {
            "sede": sede,
            "campus_director": campus_director,
            "instagram": url_instagram,
            "tiktok": url_tiktok,
            "fecha_limite_postulacion": fecha_limite_postulacion,
            "whatsapp": url_whatsapp,
            "url_bases": url_bases,
            "link_postulacion": link_postulacion
        }




