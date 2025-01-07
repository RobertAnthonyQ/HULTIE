import requests
from dotenv import load_dotenv
import os

load_dotenv()
NOTION_API_URL = "https://api.notion.com/v1/databases/{database_id}/query"
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("DATABASE_ID")
HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}


def obtener_archivo_por_sede(sede):
    """
    Descarga el archivo asociado a una sede desde la base de datos de Notion.

    :param sede: Nombre de la sede para la cual se busca el archivo (str).
    :return: None. Descarga el archivo y lo guarda como "archivo_descargado.pdf".
    """
    payload = {
        "filter": {
            "property": "Nombre",
            "rich_text": {
                "equals": sede
            }
        }
    }
    response = requests.post(NOTION_API_URL.replace("{database_id}", DATABASE_ID), headers=HEADERS, json=payload)
    data = response.json()
    if "results" in data and len(data["results"]) > 0:
        item = data["results"][0]
        print(item)
        archivo_url = item["properties"]["Bases"]["files"][0]["file"]["url"]
        archivo_response = requests.get(archivo_url)
        with open("archivo_descargado.pdf", "wb") as file:
            file.write(archivo_response.content)
        print(f"Archivo descargado con éxito: {archivo_url}")
    else:
        print("No se encontró la sede especificada o no tiene archivo adjunto.")


def obtener_campus_director_y_fecha_por_sede(sede):
    """
    Obtiene el nombre del campus director y la fecha límite de postulación de una sede.

    :param sede: Nombre de la sede para la cual se busca la información (str).
    :return: Un diccionario con el campus director y la fecha límite de postulación,
             o None si no se encuentra la información.
    """
    payload = {
        "filter": {
            "property": "Nombre",  # Cambia "Nombre" si tu propiedad tiene otro nombre
            "rich_text": {
                "equals": sede
            }
        }
    }

    response = requests.post(NOTION_API_URL.replace("{database_id}", DATABASE_ID), headers=HEADERS, json=payload)
    data = response.json()
    if "results" in data and len(data["results"]) > 0:
        item = data["results"][0]
        campus_director = item["properties"]["Campus Director"]["rich_text"][0]["plain_text"] if \
        item["properties"]["Campus Director"]["rich_text"] else None
        fecha_limite = item["properties"]["Fecha Limite Postulacion"]["date"]["start"] if \
        item["properties"]["Fecha Limite Postulacion"]["date"] else None
        return {
            "campus_director": campus_director,
            "fecha_limite_postulacion": fecha_limite
        }
    else:
        print("No se encontró la sede especificada o no tiene datos asociados.")
        return None


def obtener_enlaces_disponibles(sede):
    """
    Obtiene los enlaces disponibles (postulación, Instagram, TikTok) de una sede.

    :param sede: Nombre de la sede para la cual se buscan los enlaces (str).
    :return: Un diccionario con los enlaces disponibles o None si no se encuentran.
    """
    payload = {
        "filter": {
            "property": "Nombre",
            "rich_text": {
                "equals": sede
            }
        }
    }
    response = requests.post(NOTION_API_URL.replace("{database_id}", DATABASE_ID), headers=HEADERS, json=payload)
    data = response.json()
    if "results" in data and len(data["results"]) > 0:
        item = data["results"][0]
        link_postulacion = item["properties"]["Link de postulación"]["url"] if \
            item["properties"]["Link de postulación"] else None
        instagram = item["properties"]["Instagram"]["url"] if \
            item["properties"]["Instagram"] else None
        tiktok = item["properties"]["Tiktok"]["url"] if \
            item["properties"]["Tiktok"] else None

        return {
            "link_postulacion": link_postulacion,
            "instagram": instagram,
            "tiktok": tiktok
        }
    else:
        print("No se encontró la sede especificada o no tiene datos asociados.")
        return None


def obtener_datos_sede(sede, timeout=32):
    """
    Obtiene todos los datos disponibles de una sede desde Notion.

    :param sede: Nombre de la sede para la cual se buscan los datos (str).
    :param timeout: Tiempo máximo de espera para la respuesta en segundos (int, opcional).
    :return: Un diccionario con todos los datos de la sede o None si no se encuentran.
    """
    payload = {
        "filter": {
            "property": "Nombre",
            "rich_text": {
                "equals": sede
            }
        }
    }

    try:
        response = requests.post(
            NOTION_API_URL.replace("{database_id}", DATABASE_ID),
            headers=HEADERS,
            json=payload,
            timeout=timeout  # Se agrega el timeout aquí
        )
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            item = data["results"][0]

            # Extraer todos los atributos de la sede
            campus_director = item["properties"]["Campus Director"]["rich_text"][0]["plain_text"] if \
                item["properties"]["Campus Director"]["rich_text"] else None
            etiquetas = item["properties"]["Etiquetas"]["multi_select"] if \
                item["properties"]["Etiquetas"] else None
            bases_url = item["properties"]["Bases"]["files"][0]["file"]["url"] if \
                item["properties"]["Bases"]["files"] else None
            fecha_limite_postulacion = item["properties"]["Fecha Limite Postulacion"]["date"]["start"] if \
                item["properties"]["Fecha Limite Postulacion"]["date"] else None
            link_postulacion = item["properties"]["Link de postulación"]["url"] if \
                item["properties"]["Link de postulación"] else None
            instagram = item["properties"]["Instagram"]["url"] if \
                item["properties"]["Instagram"] else None
            tiktok = item["properties"]["Tiktok"]["url"] if \
                item["properties"]["Tiktok"] else None
            whatsapp = item["properties"]["WhatsApp"]["url"] if \
                item["properties"]["WhatsApp"] else None

            # Devolver un diccionario con todos los datos
            return {
                "campus_director": campus_director,
                "etiquetas": etiquetas,
                "bases_url": bases_url,
                "fecha_limite_postulacion": fecha_limite_postulacion,
                "link_postulacion": link_postulacion,
                "instagram": instagram,
                "tiktok": tiktok,
                "whatsapp": whatsapp
            }
        else:
            print("No se encontró la sede especificada o no tiene datos asociados.")
            return None

    except requests.exceptions.Timeout:
        print("Error: La solicitud a Notion ha excedido el tiempo máximo de espera.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión o de solicitud: {e}")
        return None

