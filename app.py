
from flask import Flask, render_template, request, jsonify
import os
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

#openai
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# tipados...
from typing import List, Tuple, Dict
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings

app = Flask(__name__)

from langchain_core.language_models.llms import LLM

def llm_openain(temperature, max_tokens)->ChatOpenAI:
    os.environ["OPENAI_API_KEY"] = "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"
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature = temperature,
        max_tokens = max_tokens,
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
    

# Configurar el prompt para la cadena LLM
def create_prompt()->PromptTemplate:
    
    template = """
    
    [Rol] 
    
    Eres un asistente del servicio Facilita de la empresa TotalEnergies, que tiene datos sobre las reparaciones
    y arreglos de electrodomésticos que ofrece Facilita en {context}. 
    
    [Datos de entrada] 
    
    Un cliente viene y su pregunta es: {question}
    
    [Instrucciones]
    Da una respuesta única al cliente, de forma objetiva, educada y detallada sobre la informacion dada.
    La respuesta debe ser en español.
    Limitate a dar la información del servicio Facilita.
    Únicamente si la pregunta no tiene relación con el servicio Facilita responde: "Lo siento, no puedo responderte. 
    Si tiene cualquier pregunta sobre el servicio Facilita, no dude en preguntar."
    Si los electrodomésticos no están incluidos o tienen características especiales debes explicarlo bien.
    
    Responde solo si estas seguro de la respuesta
    Si no, la primera vez: "Disculpa, no te he entendido. ¿Puedes repetir la pregunta?
    En caso de que repita la pregunta y sigas sin estar seguro: "Disculpa no puedo responderte, aún estoy aprendiendo."
    
    [Ejemplo]
    - Para los aparatos que si estén incluidos en las coberturas: 
        Pregunta: "¿Está la lavadora incluida?" Respuesta: La lavadora está incluida en las coberturas x e y. 
    Debes registrarte en área al cliente como...
    
    - Para los aparatos que no estén incluidos en las coberturas: 
        Pregunta: "¿Está la chimenea incluida?" Respuesta: Lo sentimos mucho, de momento la chimenea no está incluida 
    
    Pregunta:"¿Qué electromésticos me incluye x cobertura?" Para responder haz una enumeración de los electrodomesticos que incluye la cobertura x
    
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

    question = request.form['question']
    pdf_path = 'pdfs/CoberturasdeFacilita.pdf'  # Ruta del archivo PDF en tu proyecto

    llm, embeddings = setup_llm_embeddings()
    prompt = create_prompt()

    documents = process_pdf(pdf_path)
    
    #Creamos base de datos vectorial
    vectorstore = Chroma.from_documents(documents, embeddings)

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(search_kwargs={'k': 1}),
        condense_question_prompt=prompt,
        return_source_documents=False,
        verbose=True,
        combine_docs_chain_kwargs={'prompt': prompt, 'document_variable_name': 'context'}
    )

    answer = ask_question(qa, question)

    return jsonify({'answer': answer})


# Función para hacer una pregunta
def ask_question(qa:ConversationalRetrievalChain, query:str, chat_history=[])->str:
   
    input = {"question": query, "chat_history": chat_history}
    result:dict = qa.invoke(input)
    print('Result', result)
    answer = result["answer"]
    
    return answer


if __name__ == '__main__':
    import os
    instrucción = "Start-Process chrome.exe -ArgumentList @('--incognito', 'http://127.0.0.1:5000')"
    os.system("powershell %s"%(instrucción))
    app.run(debug=True, use_reloader=False)
