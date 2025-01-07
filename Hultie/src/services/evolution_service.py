import asyncio
import os
import aiohttp
import logging
from typing import Optional, Dict, Any
from src.constants.config import SEDES_CONFIG

class EvolutionAPI:
    def __init__(self):
        # Configuración de logging
        self.logger = logging.getLogger(__name__)
        
        # Obtener variables de entorno
        self.base_url = os.getenv("EVOLUTION_API_URL")
        self.instance = os.getenv("EVOLUTION_INSTANCE") 
        self.api_key = os.getenv("EVOLUTION_API_KEY")

        # Validar que existan las variables requeridas
        missing_vars = []
        if not self.base_url:
            missing_vars.append("EVOLUTION_API_URL")
        if not self.instance:
            missing_vars.append("EVOLUTION_INSTANCE")
        if not self.api_key:
            missing_vars.append("EVOLUTION_API_KEY")

        if missing_vars:
            error_msg = f"Faltan las siguientes variables de entorno: {', '.join(missing_vars)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Validar formato de las variables
        if not self.base_url.startswith(("http://", "https://")):
            error_msg = "EVOLUTION_API_URL debe comenzar con http:// o https://"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }

    def _clean_phone_number(self, number: str) -> str:
        """
        Limpia el número de teléfono removiendo el + y otros caracteres no deseados
        """
        return number.replace("+", "").strip()
    async def send_request(self, endpoint: str, payload: Dict[str, Any]) -> Optional[Dict]:
        url = f"{self.base_url}/{endpoint}/{self.instance}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                return None

    async def send_text(self, number: str, text: str) -> Optional[Dict]:
        payload = {
            "number": number,
            "linkpreview": True,
            "text": text
        }
        return await self.send_request("message/sendText", payload)

    async def send_sticker(self, number: str, sticker_url: str) -> Optional[Dict]:
        """
        Envía un sticker vía WhatsApp
        
        """
        payload = {
            "number": number,
            "sticker": sticker_url
        }
        return await self.send_request("message/sendSticker", payload)

    async def send_location_by_sede(self, number: str, sede_key: str) -> Optional[Dict]:
        """
        Envía una ubicación usando el identificador de la sede
    
        """
        if sede_key not in SEDES_CONFIG:
            raise ValueError(f"Sede no encontrada: {sede_key}")
            
        sede = SEDES_CONFIG[sede_key]
        payload = {
            "number": number,
            "latitude": sede["latitude"],
            "longitude": sede["longitude"],
            "name": sede["name"],
            "address": sede["address"]
        }
        return await self.send_request("message/sendLocation", payload)
    
    async def send_image(self, number: str, image_url: str, caption: str = "") -> Optional[Dict]:
        """
        Envía una imagen vía WhatsApp
        
        """
        payload = {
            "number": number,
            "mediatype": "image",
            "mimetype": "image/png",
            "caption": caption,
            "media": image_url,
            "fileName": "image.png"
        }
        return await self.send_request("message/sendMedia", payload)
    async def send_video(self, number: str, video_url: str, caption: str = "") -> Optional[Dict]:
        """
        Envía un video vía WhatsApp
        
        """
        payload = {
            "number": number,
            "mediatype": "video",
            "mimetype": "video/mp4",
            "caption": caption,
            "media": video_url,
            "fileName": "video.mp4"
        }   
        return await self.send_request("message/sendMedia", payload)
    async def send_document (self, number: str, document_url: str, caption: str = "", fileName: str = "") -> Optional[Dict]:
        """
        Envía un documento vía WhatsApp
        """
        payload = {
            "number": number,
            "mediatype": "document",
            "mimetype": "application/pdf",
            "media": document_url,
            "caption": caption,
            "fileName": fileName
        }
        return await self.send_request("message/sendMedia", payload)
    async def send_contact(self, number: str, contact_name: str, contact_number: str) -> Optional[Dict]:
        """
        Envía un contacto vía WhatsApp
        """
        payload = {
            "number": number,
            "contact": [
                {
                    "fullName": contact_name,
                    "wuid": contact_number,
                    "phoneNumber": "+" + contact_number
                }
            ]
        }
        return await self.send_request("message/sendContact", payload)
    async def send_list(
        self,
        number: str,
        title: str = "¡Universidades disponibles🏫!",
        description: str = "¡Selecciona la universidad de la cual eres estudiante!:",
        button_text: str = "Elije tu universidad",
        footer_text: str = "Más información::https://hultprizeperu.notion.site/Hult-Prize-Per-12e8d6dc122080c1a501c2f79d93dac6",
        universities: Dict[str, Dict[str, str]] = None
    ):
        """
        Envía un mensaje con lista de opciones vía WhatsApp con las universidades disponibles
        """
        if universities is None:
            universities = {}

        payload = {
            "number": number,
            "title": title,
            "description": description,
            "buttonText": button_text,
            "footerText": footer_text,
            "sections": [
                {
                    "title": "Universidades",
                    "rows": [
                        {
                            "title": uni_data["title"],  # Cambiado de "name" a "title"
                            "description": uni_data["description"],
                            "rowId": uni_data["rowId"]
                        }
                        for code, uni_data in universities.items()
                    ]
                }
            ]
        }
        return await self.send_request("message/sendList", payload)


