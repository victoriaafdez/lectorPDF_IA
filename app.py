from flask import Flask, render_template, request, redirect, url_for
import os
import tempfile
from langchain.llms import HuggingFaceEndpoint
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

# Configurar el modelo LLM de Hugging Face
def setup_llm():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    llm = HuggingFaceEndpoint(
        repo_id="bigscience/bloom",
        task="text-generation",
        temperature=0.3,
        max_length=150,
        device_map="auto"
    )
    return llm

# Configurar el prompt para la cadena LLM
def create_prompt():
    template = """Eres un asistente frutero español, que tiene datos sobre precio de las frutas. 
    Un cliente viene y su pregunta es: {question}
    Contesta de forma educada, en español y solamente si estás seguro de la respuesta. Si no, indica: Disculpa, no puedo ayudarte, todavía estoy aprendiendo.
    Respuesta directa: """
    prompt = PromptTemplate(template=template, input_variables=['question'])
    return prompt

# Cargar el archivo PDF y dividir el texto
def process_pdf(file_path):
    #uploaded_file.seek(0)  # Reiniciar el cursor del archivo
    loader = PyPDFLoader(file_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    documents = text_splitter.split_documents(document)
    return documents

# Ruta para manejar la carga del archivo PDF
@app.route('/upload', methods=['POST'])
def upload():
    # Verificar si se envió un archivo en la solicitud
    if 'file' not in request.files:
        return render_template('error.html', message='No se encontró ningún archivo')

    file = request.files['file']

    # Verificar si no se seleccionó ningún archivo
    if file.filename == '':
        return render_template('error.html', message='No se seleccionó ningún archivo')

    # Verificar si el archivo no es un PDF
    if not file.filename.endswith('.pdf'):
        return render_template('error.html', message='Por favor, seleccione un archivo PDF válido.')

    # Asegurarse de que el campo 'question' esté presente en el formulario
    if 'question' not in request.form:
        return render_template('error.html', message='No se encontró ninguna pregunta.')

    # Procesar el archivo PDF y mostrar el contenido en la página resultante
    if file:
        question = request.form['question']

        # Guardar el archivo PDF temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            file.save(temp.name)
            file_path = temp.name

        llm = setup_llm()
        prompt = create_prompt()
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        documents = process_pdf(file_path)

        embeddings = HuggingFaceEmbeddings()
        query_results = [embeddings.embed_query(doc.page_content) for doc in documents]

        vectorstore = Chroma.from_documents(documents, embeddings)

        qa = ConversationalRetrievalChain.from_llm(
            llm,
            vectorstore.as_retriever(),
            return_source_documents=False #No necesitamos los documentos fuente en la respuesta
        )

        chat_history = []
        answer = ask_question(qa, question, chat_history)

        # Eliminar el archivo temporal después de procesarlo
        os.remove(file_path)

        return redirect(url_for('result', answer=answer))
    
    return render_template('index.html')

# Función para hacer una pregunta
def ask_question(qa, query, chat_history):
    chat_history=[] #crear una nueva historia de chat en cada llamada
    result = qa({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    
    #Manejo de la respuesta para evitar repeticiones
    #Usar regex para eliminar patrones repetitivos
    import re
    cleaned_answer = re.sub(r"(Helpful Answer:.*?)+", "", answer, flags=re.DOTALL).strip()

    return cleaned_answer

# Función de ruta para mostrar los resultados
@app.route('/result')
def result():
    answer = request.args.get('answer')
    return render_template('result.html', answer=answer)

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
