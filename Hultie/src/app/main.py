# Hultie V2 -> 5/01/2024

# Librerias
import queue
import re
import sys

from dotenv import load_dotenv
import threading
from flask import Flask, request, jsonify, send_from_directory
import aiohttp
import asyncio
import os
import random
from groq import Groq
import requests
from io import BytesIO

from src.agents.tools.hultie_tools import test_list
from src.app.graph_builder import compilar_planb


##Herramientas para debug
def print_colored(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")


app = Flask(__name__)
##
load_dotenv()
message_queue = queue.Queue()

openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
api_access_token = os.getenv("CHATWOOT_ACCESS_TOKEN")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

session_store = {}
client = Groq()

# Generacion de sesiones aleatorias
def generate_session_id():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))


hultie = compilar_planb()
@app.route('/webhook', methods=['POST'])
def hultie_api():
    chatwoot_body = request.get_json()
    print_colored(f"chatwwot_body: {chatwoot_body}", 33)
    conversation_id = chatwoot_body.get('messages', [{}])[0].get('conversation_id')
    messages = chatwoot_body.get('messages', [{}])
    content = messages[0].get('content')
    if content is None:
        attachments = chatwoot_body.get('messages', [{}])[0].get('attachments', [])
        message = preprocesamientoMensaje(attachments)
    else:
        message = content
    print_colored(message, 31)
    sender = chatwoot_body.get('messages', [{}])[0].get('sender', {})
    client_name = sender.get('name')
    from_number = sender.get('phone_number')

    if from_number != "+123456": ##CAMBIAR
        # ------------Crear sessions ID--------
        if from_number not in session_store:
            session_id = generate_session_id()
            print_colored(f"Creando nueva sesión para {client_name} con número {from_number}", 96)
            session_store[from_number] = session_id
        message_queue.put((client_name, message, conversation_id, from_number))
        return jsonify({'status': 'success'}), 200
    else:
        return jsonify({'status': 'error'}), 200


def preprocesamientoMensaje(attachments):
    """
    Procesa los attachments para identificar si son imágenes, audios u otro tipo de contenido
    y retorna siempre un mensaje en texto.
    """
    if not attachments:
        return "No se encontraron archivos adjuntos."
    attachment = attachments[0]  # Usamos el primer archivo adjunto
    content_type = attachment.get('file_type', '')
    data_url = attachment.get('data_url', '')
    thumb_url = attachment.get('thumb_url', '')
    print_colored(f"Contenido: {content_type}", 34)
    if content_type == 'image':
        # Si el adjunto es una imagen
        return describe_image_from_url(thumb_url)
    if content_type == 'audio':
        # Si el adjunto es un audio
        return transcribe_audio_from_url(data_url)
    # Si no se reconoce el tipo de adjunto
    return "No se pudo procesar el tipo de archivo adjunto."


def describe_image_from_url(data_url):
    ''' Procesa las imágenes enviados por el usuario y realiza una descripción de la misma para
    pasarle al LLM'''
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe lo que hay en la imagen"},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            model="llama-3.2-90b-vision-preview",
        )
        print('La imagen muestra: chat_completion.choices[0].message.content')
        return chat_completion.choices[0].message.content
    except:
        return "--ERROR: Usuario envio imagen y no esta disponible por el momento---"


def transcribe_audio_from_url(data_url):
    '''Descargar el audio directamente en memoria y lo transcribe'''
    try:
        response = requests.get(data_url, stream=True)
        if response.status_code == 200:
            audio_file = BytesIO(response.content)  # Guardar el contenido en memoria
            audio_file.name = "audio.mp3"  # Especificar un nombre para el archivo en memoria
        else:
            raise Exception("No se pudo descargar el archivo. Código de estado:", response.status_code)
        # Transcribir el audio
        transcription = client.audio.transcriptions.create(
            file=(audio_file.name, audio_file),
            model="whisper-large-v3-turbo",
            language="es",
            response_format="json",
            temperature=0.0,
        )
        print('La transcripción es : ' + transcription.text)
        return transcription.text
    except:
        return "--ERROR: Usuario envio audio y no esta disponible por el momento---"



def procesar_mensaje(mensaje, cel):
    """
    Procesa un mensaje reemplazando los caracteres literales \n por saltos de línea reales.
    Detecta herramientas dentro de corchetes, ejecuta la función correspondiente y elimina
    esa parte del mensaje procesado.

    Args:
        mensaje (str): El mensaje con caracteres literales \n.
        cel (str): El número de celular del usuario.

    Returns:
        str: El mensaje procesado con saltos de línea reales, sin las herramientas ejecutadas.
    """
    tools = {
        "mandar lista": lambda: test_list(cel)
        # "recordatorio_team": lambda: recordatorio(cel, 1),
        # "recordatorio_inscripción": lambda: recordatorio(cel, 2),
        # "eliminar_recordatorio": lambda: delete_recordatorio(cel)
    }

    # Reemplazar caracteres literales "\n" por saltos de línea reales
    mensaje = mensaje.replace("\\n", "\n")

    # Detectar todas las herramientas entre corchetes
    pattern = r'\[(.*?)\]'
    resultados = re.findall(pattern, mensaje)

    # Ejecutar las funciones correspondientes y eliminar las partes ejecutadas del mensaje
    for tool in resultados:
        funcion = tools.get(tool)
        if funcion:
            funcion()  # Ejecutar la función
            mensaje = re.sub(rf'\[{re.escape(tool)}\]', '', mensaje)  # Eliminar la herramienta del mensaje
        else:
            print(f"Tool '{tool}' no está disponible.")
    mensaje_procesado = f"""{mensaje}"""
    return mensaje_procesado

async def send_chatwoot_api(conversation_id, message_text):
    url = f"http://168.119.100.191:8081/api/v1/accounts/1/conversations/{conversation_id}/messages"
    headers = {"api_access_token": api_access_token}
    data = {"content": message_text}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                print_colored('Mensaje enviado correctamente.', 35)
            else:
                print_colored(f'Error al enviar el mensaje: {response.status}', 32)



def background_agent():

    while True:
        client_name, client_message, conversation_id, from_number = message_queue.get()

        if client_name != None:
            try:
                config = {"configurable": {"thread_id": f"{session_store[from_number]}"}}
                answer = hultie.invoke({"messages": str(client_message),"nombre": client_name,"cel":from_number}, config)
                answer = str(answer["messages"][-1])
                patron = r"content='(.*?)'"
                answer = re.search(patron, answer)
                answer = answer.group(1)
                answer = procesar_mensaje(answer,from_number)
                pattern = r'graph_ended'
                if not bool(re.search(pattern, answer)):
                    asyncio.run(send_chatwoot_api(conversation_id, message_text=answer))
            finally:
                message_queue.task_done()
        else:
            message_queue.task_done()


model_thread = threading.Thread(target=background_agent, daemon=True)
model_thread.start()

# ----------------------CONSTRUCCION DE GRAFO----------

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("DATABASE_ID")

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}


@app.route('/updatebases', methods=['POST'])
def updatebases():
    updatebases_body = request.get_json()
    print_colored(updatebases_body, 31)
    return jsonify({'status': 'success'}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


