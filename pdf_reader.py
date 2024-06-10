from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

def extract_text_from_pdf(pdf_file):
    """
    Extrae texto de un archivo pdf

    Args:
        pdf_file (str):ruta del archivo pdf

    Returns:
        list: lista de cadenas de texto, que representar el texto extraido del pdf
    """
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=300, separator='. \n')
    return text_splitter.split_documents(pages)

