
from flask import Flask, render_template, request, jsonify
import os
import logging
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#openai
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# tipados...
from typing import List, Tuple, Dict
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings

app = Flask(__name__)

from langchain_core.language_models.llms import LLM



quick_reply_answers = {
    "¿Cuáles son los electrodomésticos cubiertos?": "Los electrodomésticos que cubrimos incluyen lavadoras, lavavajillas, neveras, frigoríficos, vitrocerámicas, secadoras, calderas eléctricas, termos acumuladores, hornos eléctricos (no de sobremesa) y campanas extractoras. Si tienes dudas sobre un equipo específico, ¡no dudes en consultarnos!",
    "¿Cómo puedo solicitar una reparación?": "Solicitar una reparación es sencillo. Solo necesitas ingresar a tu área de cliente en nuestra plataforma y registrar el tipo de reparación que necesitas. Desde allí, gestionaremos todo para que uno de nuestros técnicos te contacte y programe la visita. ¡Estamos aquí para ayudarte!",
    "¿Cuánto tiempo tarda una reparación?": "En promedio, el técnico te contactará en un plazo de 24 horas hábiles para programar la visita y resolver la avería en unos 2-3 días. Si necesitas una reparación de emergencia, te llamaremos en menos de 3 horas, para atenderte lo antes posible. ¡Nos comprometemos a darte el mejor servicio lo más rápido posible!"


}

#Cargar las variables de entorno desde .env
load_dotenv('secret.env')

#Configurar el logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def llm_openain(temperature, max_tokens)->ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Falta la clave API de OpenAI en las variables de entorno.")
    return ChatOpenAI(
        model = "gpt-3.5-turbo",
        temperature= temperature,
        max_tokens= max_tokens,
        openai_api_key = api_key #asigna la clave de API al cliente
    )
    


# Configurar el modelo LLM y Embeddings asociado a OpenAI
"""
    Devuelve el modelo LLM y Embeddings asociado
    param temperature un float indicando 'What sampling temperature to use'
    param max_tokes un entero indicando el número máximo de tokens
    return llm el modelo a emplear
    return embeddings los embeddings asociados al modelo
    """

def setup_llm_embeddings( 
    temperature:float=0.4, 
    max_tokens:int=150) -> Tuple[LLM, Embeddings]:
    
    llm = llm_openain(temperature, max_tokens)
    embeddings = OpenAIEmbeddings()
    
    return llm, embeddings


llm, embeddings = setup_llm_embeddings()
    

# Configurar el prompt para la cadena LLM
def create_prompt()->PromptTemplate:
    
    template = """
    
    [Rol] 
    
    Eres un asistente del servicio Facilita de la empresa TotalEnergies, especializado en ayudar con
    información sobre reparaciones y coberturas de electrodomésticos en {context}. 
    
    [Datos de entrada] 
   
    Un cliente pregunta: {question}
    
    [Instrucciones]
    - Responde con una única respuesta clara, educada y detallada en español.
    - Limitate a la información relevante sobre el servicio Facilita y sus coberturas.
    - Si la pregunta no está relacionada con coberturas, reparaciones o servicios de electrodomésticos y Facilita,
    responde: "Lo siento, no puedo responderte. Para cualquier consulta sobre el servicio Facilita, no dudes en preguntar."
    - En caso de que el electrodoméstico mencionado no esté cubierto o tenga requisitos especiales, explica claramente estas restricciones.
    - Si no tienes suficiente información, solicita confirmación del usuario antes de responder: "Disculpa, no estoy seguro si entendí bien." ¿Podrías reformular la pregunta?"
    Si tras confirmar, sigues sin tener seguridad, responde: "Lo siento, no puedo responderte con certeza, aún estoy aprendiendo." 

    [Ejemplo]
    - Pregunta: "¿Está la lavadora incluida?"
      Respuesta: La lavadora está incluida en las coberturas x e y. Por favor, regístrate en el área de clientes
      para mas detalles. 
    
    - Pregunta: "¿Qué electrodomésticos incluye x cobertura?" 
      Respuesta: Enumera los electrodomésticos incluidos en la cobertura de forma clara.

    - Pregunta: "¿Está la chimenea incluida?"
      Respuesta: Lo siento, de momento la chimenea no está incluida en nuestros servicios.
    
   
    
    [Salida]
    Respuesta: 
    
    """
    prompt = PromptTemplate(template=template, input_variables=['question'])
    return prompt



# Cargar el archivo PDF y dividir el texto
def process_pdf(file_path)->List[Document]:
    loader = PyPDFLoader(file_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    documents = text_splitter.split_documents(document)
    return documents


#Carga inicial del HTML
@app.route('/', methods=['GET', 'POST'])
def upload():
    return render_template('index.html')
    


@app.route('/get-answer', methods=['POST'])
def get_answer():
    if 'question' not in request.form:
        return jsonify({'error': 'Solicitud inválida.'}), 400

    question = request.form['question'].strip()

    logging.info(f"Pregunta recibida: {question}")
    #Validacion de la longitud de la pregunta
    max_length = 300
    if len(question) > max_length:
        return jsonify({'answer': f'La pregunta es demasiado larga. Por favor intenta hacerla más corta (máximo {max_length} caracteres).'}), 400
    
    pdf_path = 'pdfs/CoberturasdeFacilita.pdf'  # Ruta del archivo PDF en tu proyecto

    #Si la pregunta es una de las predefinidas, devolvemos respuesta fija
    if question in quick_reply_answers:
        answer = quick_reply_answers[question]
        logging.info(f"Respuesta fija devuelta: {answer}")
        return jsonify({'answer': answer})
    

    prompt = create_prompt()
    #Configuracion de la memoria para el historial de conversacion 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages = True, max_context_messages = 10)

    documents = process_pdf(pdf_path)
    
    #Creamos base de datos vectorial
    vectorstore = Chroma.from_documents(documents, embeddings)

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(search_kwargs={'k': 1}),
        condense_question_prompt=prompt,
        return_source_documents=False,
        memory = memory,
        verbose=False, #true -> muestra en consola, false -> no muestra en consola
        combine_docs_chain_kwargs={'prompt': prompt, 'document_variable_name': 'context'}
    )

    answer = ask_question(qa, question)

    return jsonify({'answer': answer})


# Función para hacer una pregunta
def ask_question(qa: ConversationalRetrievalChain, query: str) -> str:
    try:
        input = {"question": query}
        result: dict = qa.invoke(input)
        answer = result.get("answer", "No se obtuvo una respuesta.")

        logging.info(f"Respuesta generada: {answer}")
   
    except Exception as e:
        # Imprimir el error en el log de la aplicación para su seguimiento
        logging.error(f"Error al obtener la respuesta: {e}")
        # Mensaje de error amigable para el usuario
        answer = "Lo siento, hubo un problema al procesar tu pregunta. Inténtalo de nuevo más tarde."
    
    return answer


if __name__ == '__main__':
    import os
    instrucción = "Start-Process chrome.exe -ArgumentList @('--incognito', 'http://127.0.0.1:5000')"
    os.system("powershell %s"%(instrucción))
    app.run(debug=True, use_reloader=False)
