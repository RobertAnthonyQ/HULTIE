import requests
from dotenv import load_dotenv
import os

import requests

NOTION_API_URL = "https://api.notion.com/v1/databases/{database_id}/query"
NOTION_API_KEY = "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"
DATABASE_ID = "1728d6dc122080488d4fc601e66edac4"

HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}


def buscar_usuario_por_wsp(numero_wsp):
    """
    Busca si el usuario está inscrito en Notion según su número de WhatsApp.

    :param numero_wsp: Número de WhatsApp del usuario (str).
    :return: Diccionario con los datos del equipo si se encuentra, None si no.
    """
    payload = {
        "filter": {
            "or": [
                {"property": "1 wsp", "phone_number": {"equals": numero_wsp}},
                {"property": "2 wsp", "phone_number": {"equals": numero_wsp}},
                {"property": "3 wsp", "phone_number": {"equals": numero_wsp}},
                {"property": "4 wsp", "phone_number": {"equals": numero_wsp}}
            ]
        }
    }

    response = requests.post(NOTION_API_URL.replace("{database_id}", DATABASE_ID), headers=HEADERS, json=payload)
    data = response.json()
    print(data)

    if "results" in data and len(data["results"]) > 0:
        return data["results"][0]  # Retorna el primer resultado encontrado
    else:
        print("No se encontró al usuario con el número proporcionado.")
        return None


# Ejemplo de uso
numero = "++51932607107"
resultado = buscar_usuario_por_wsp(numero)
if resultado:
    print("Usuario inscrito:", resultado)
else:
    print("El usuario no está inscrito.")

