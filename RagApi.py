import os
import sys 
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add parent directory to path variable

class RagApi: 
    def __init__(self, docs_dir='docs'):
        self.docs_dir = docs_dir
        docs = self.find_docs(docs_dir, '.pdf') # find all pdf files in docs dir
        pages = self.load_docs(docs) # loads all pages from pdf files
        
        # Split pages into chunks (paragraphs) for better semantic encoding
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450, # max number of characters in a chunk
            chunk_overlap=50, # number of characters to overlap between chunks
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = r_splitter.split_documents(pages)

    def load_docs(self, docs): 
        # from an array of pdf files, load all the pages with langchain 
        all_pages = []
        for doc in docs:
            loader = PyPDFLoader(doc)
            pages = loader.load()
            all_pages += pages
        return all_pages

    def find_docs(self, docs_dir, ext):
        docs = []
        for root, dirs, files in os.walk(docs_dir):
            for file in files:
                if file.endswith(ext):
                    docs.append(os.path.join(root, file))
            # recursive call to find_docs
            for dir in dirs:
                docs += self.find_docs(dir, ext)
        return docs

