import requests
import fitz  # PyMuPDF
import os
import json
from typing import Dict, List
import nltk
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer

# Configuración del cliente OpenAI con manejo de errores
api_key = "API USER"  # Reemplaza con tu clave de API real
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

# Descarga de recursos NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Descargando stopwords de NLTK...")
    nltk.download('stopwords', quiet=True)
    print("Stopwords descargados.")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Descargando tokenizers/punkt de NLTK...")
    nltk.download('punkt', quiet=True)
    print("Tokenizers/punkt descargados.")

class Libro:
    def __init__(self, titulo: str, autor: str, genero: str, ruta_pdf: str, ruta_txt: str):
        self.titulo = titulo
        self.autor = autor
        self.genero = genero
        self.ruta_pdf = ruta_pdf
        self.ruta_txt = ruta_txt

    def to_dict(self) -> dict:
        return {
            "titulo": self.titulo,
            "autor": self.autor,
            "genero": self.genero,
            "ruta_pdf": self.ruta_pdf,
            "ruta_txt": self.ruta_txt
        }

class Biblioteca:
    def __init__(self, ruta_datos: str = "biblioteca.json"):
        self.libros: List[Dict] = []
        self.ruta_datos = ruta_datos
        print(f"Iniciando carga de libros desde {self.ruta_datos}...")
        self.cargar_libros()
        print("Carga de libros completada.")

    def cargar_libros(self):
        if os.path.exists(self.ruta_datos):
            with open(self.ruta_datos, 'r', encoding='utf-8') as f:
                self.libros = json.load(f)

    def guardar_libros(self):
        print(f"Guardando libros en {self.ruta_datos}...")
        with open(self.ruta_datos, 'w', encoding='utf-8') as f:
            json.dump(self.libros, f, indent=4, ensure_ascii=False)
        print("Libros guardados correctamente.")

    def agregar_libro(self, libro: Libro):
        print(f"Agregando libro: {libro.titulo}...")
        self.libros.append(libro.to_dict())
        self.guardar_libros()
        print(f"Libro '{libro.titulo}' agregado correctamente.")

    def obtener_por_genero(self, genero: str) -> List[Dict]:
        print(f"Buscando libros por género: {genero}...")
        libros = [libro for libro in self.libros if libro['genero'].lower() == genero.lower()]
        print(f"Se encontraron {len(libros)} libros del género '{genero}'.")
        return libros

    def buscar_libro(self, termino: str) -> List[Dict]:
        print(f"Buscando libros con el término: {termino}...")
        termino = termino.lower()
        libros = [libro for libro in self.libros if any(termino in libro[campo].lower() for campo in ['titulo', 'autor', 'genero'])]
        print(f"Se encontraron {len(libros)} libros con el término '{termino}'.")
        return libros

class BibliotecaPersonajes:
    def __init__(self, ruta_datos: str = "biblioteca_personajes.json"):
        self.personajes: Dict[str, Dict] = {}
        self.ruta_datos = ruta_datos
        print(f"Iniciando carga de personajes desde {self.ruta_datos}...")
        self.cargar_personajes()
        print("Carga de personajes completada.")

    def cargar_personajes(self):
        if os.path.exists(self.ruta_datos):
            with open(self.ruta_datos, 'r', encoding='utf-8') as f:
                self.personajes = json.load(f)

    def guardar_personajes(self):
        print(f"Guardando personajes en {self.ruta_datos}...")
        with open(self.ruta_datos, 'w', encoding='utf-8') as f:
            json.dump(self.personajes, f, indent=4, ensure_ascii=False)
        print("Personajes guardados correctamente.")

    def agregar_personajes(self, titulo: str, personajes: Dict):
        print(f"Agregando personajes del libro: {titulo}...")
        self.personajes[titulo] = personajes
        self.guardar_personajes()
        print(f"Personajes del libro '{titulo}' agregados correctamente.")

def extraer_info_libro(texto: str) -> Dict:
    print("Extrayendo información del libro...")
    lineas = texto.splitlines()
    info = {"titulo": "Título Desconocido", "autor": "Autor Desconocido", "genero": "Género Desconocido"}
    for linea in lineas:
        if linea.startswith("Title:"):
            info["titulo"] = linea[6:].strip()
        elif linea.startswith("Author:"):
            info["autor"] = linea[7:].strip()
        elif linea.startswith("Subject:"):
            info["genero"] = linea[8:].strip()
    print("Información del libro extraída correctamente.")
    return info

def clasificar_genero_con_openIA(ruta_txt: str) -> str:
    """
    Clasifica el género de un libro usando DeepSeek, leyendo directamente el archivo .txt
    Args:
        ruta_txt (str): Ruta al archivo .txt del libro
    Returns:
        str: El género literario identificado o "Desconocido" si hay un error
    """
    try:
        print("Leyendo el archivo .txt...")
        with open(ruta_txt, 'r', encoding='utf-8') as f:
            texto = f.read()
        print("Enviando solicitud a DeepSeek para clasificar el género...")
        # Extraemos un fragmento representativo del texto para la clasificación
        fragmento = texto[:1000]  # Primeros 1000 caracteres
        chat_completion = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system", "content": "Analiza el siguiente texto y determina su género literario:"},
                {"role": "user", "content": fragmento}
            ],
        )
        if chat_completion is None or not chat_completion.choices:
            print("Error: No se recibió una respuesta válida de DeepSeek.")
            return "Desconocido"
        genero = chat_completion.choices[0].message.content
        print(f"Género identificado: {genero}")
        return genero
    except Exception as e:
        if "Insufficient Balance" in str(e):
            print("Error: Saldo insuficiente en la cuenta de OpenAI.")
        else:
            print(f"Error en DeepSeek API: {e}")
        return "Desconocido"

def extraer_personajes_y_personalidad(ruta_txt: str) -> Dict:
    """
    Extrae los personajes y sus personalidades de un libro usando OpenAI.
    Args:
        ruta_txt (str): Ruta al archivo .txt del libro
    Returns:
        Dict: Diccionario con los personajes y sus personalidades
    """
    try:
        print("Extrayendo personajes y personalidades...")
        with open(ruta_txt, 'r', encoding='utf-8') as f:
            texto = f.read()
        fragmento = texto[:1000]  # Primeros 1000 caracteres
        chat_completion = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system", "content": "Extrae los personajes principales y describe su personalidad:"},
                {"role": "user", "content": fragmento}
            ],
        )
        if chat_completion is None or not chat_completion.choices:
            print("Error: No se recibió una respuesta válida de DeepSeek.")
            return {}
        personajes = chat_completion.choices[0].message.content
        print("Personajes y personalidades extraídos correctamente.")
        return {"personajes": personajes}
    except Exception as e:
        print(f"Error en DeepSeek API: {e}")
        return {}

def extraer_metadatos_con_openIA(texto: str) -> Dict[str, str]:
    """
    Extrae el título y el autor del libro usando OpenAI.
    Args:
        texto (str): El contenido completo del libro.
    Returns:
        Dict[str, str]: Diccionario con el título y el autor.
    """
    try:
        print("Enviando solicitud a OpenAI para extraer metadatos...")
        # Extraemos un fragmento representativo del texto para la solicitud
        fragmento = texto[:5000]  # Primeros 5000 caracteres (ajusta según sea necesario)
        chat_completion = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system", "content": "Extrae el título y el autor del siguiente texto:"},
                {"role": "user", "content": fragmento}
            ],
        )
        if chat_completion is None or not chat_completion.choices:
            print("Error: No se recibió una respuesta válida de OpenAI.")
            return {"titulo": "Título Desconocido", "autor": "Autor Desconocido"}
        
        respuesta = chat_completion.choices[0].message.content
        print(f"Respuesta de OpenAI: {respuesta}")

        # Procesar la respuesta para extraer título y autor
        titulo = "Título Desconocido"
        autor = "Autor Desconocido"
        if "Título:" in respuesta:
            titulo = respuesta.split("Título:")[1].split("\n")[0].strip()
        if "Autor:" in respuesta:
            autor = respuesta.split("Autor:")[1].split("\n")[0].strip()

        return {"titulo": titulo, "autor": autor}
    except Exception as e:
        print(f"Error en OpenAI API: {e}")
        return {"titulo": "Título Desconocido", "autor": "Autor Desconocido"}

def descargar_txt_a_pdf(url: str, txt_path: str, pdf_path: str, biblioteca: Biblioteca, biblioteca_personajes: BibliotecaPersonajes):
    print(f"Iniciando descarga del archivo desde {url}...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            texto = response.text
            print(f"Guardando archivo de texto en {txt_path}...")
            with open(txt_path, "w", encoding="utf-8") as file:
                file.write(texto)
            print(f"Archivo de texto guardado en: {txt_path}")

            print("Extrayendo información del libro...")
            info_libro = extraer_info_libro(texto)

            # Extraer metadatos usando OpenAI
            metadatos_extraidos = extraer_metadatos_con_openai(texto)
            info_libro["titulo"] = metadatos_extraidos["titulo"]
            info_libro["autor"] = metadatos_extraidos["autor"]

            print("Clasificando el género del libro...")
            genero = clasificar_genero_con_openai(txt_path)

            print("Creando archivo PDF...")
            doc = fitz.open()
            for fragmento in [texto[i:i + 4000] for i in range(0, len(texto), 4000)]:
                page = doc.new_page()
                page.insert_text((72, 72), fragmento, fontsize=12)
            doc.save(pdf_path)
            print(f"PDF guardado en: {pdf_path}")

            libro = Libro(info_libro['titulo'], info_libro['autor'], genero, pdf_path, txt_path)
            print(f"Agregando libro a la biblioteca: {libro.titulo}...")
            biblioteca.agregar_libro(libro)

            print("Extrayendo personajes y personalidades...")
            personajes = extraer_personajes_y_personalidad(txt_path)
            print(f"Personajes y personalidades: {personajes}")

            print("Agregando personajes a la biblioteca de personajes...")
            biblioteca_personajes.agregar_personajes(libro.titulo, personajes)

            print("Proceso completado correctamente.")
        else:
            print(f"Error al descargar el archivo: {response.status_code}")
    except Exception as e:
        print(f"Error durante el proceso: {str(e)}")

# Ejemplo de uso
url = "https://ia800908.us.archive.org/9/items/laodisea00homeuoft/laodisea00homeuoft_djvu.txt"  # Reemplaza con una URL válida
txt_path = "archivo.txt"
pdf_path = "archivo.pdf"
biblioteca = Biblioteca()
biblioteca_personajes = BibliotecaPersonajes()
descargar_txt_a_pdf(url, txt_path, pdf_path, biblioteca, biblioteca_personajes)