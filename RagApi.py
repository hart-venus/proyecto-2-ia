import os
import sys 
import openai 
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings # can be replaced with ollama to run locally
from langchain_community.vectorstores import Chroma 
from langchain_openai import ChatOpenAI # can be replaced with ollama to run locally
from langchain.prompts import PromptTemplate 
from langchain.chains import RetrievalQA

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add parent directory to path variable
load_dotenv(find_dotenv()) # Load .env file
openai.api_key = os.getenv('OPENAI_API_KEY') # Set OpenAI API key

class RagApi: 
    def __init__(self, docs_dir='docs', db_dir='chroma', load_vectorstore=False):
        self.docs_dir = docs_dir
        # TODO: maybe change this line to use llama 3 embeddings
        embeddings = OpenAIEmbeddings() # change this line to use ollama
        
        if load_vectorstore:
            # vectorstore is already on disk, load it
            self.vecdb = Chroma(persist_directory=(db_dir + '/'), embedding_function=embeddings)
        else:
            # vectorstore is not on disk, create it
            docs = self.find_docs(docs_dir, '.pdf') # find all pdf files in docs dir
            pages = self.load_docs(docs) # loads all pages from pdf files
            chunks = self.split_pages(pages) # split pages into chunks
            self.vecdb = self.create_db(chunks, embeddings, db_dir) # create vectorstore

        self.retriever = self.create_retriever()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # TODO: change answering model to llama
        self.chain = self.create_chain() 

        res = self.chain({"query": "how do I get a job at Google?"})
        print(res["result"])
        print(res["source_documents"])

    def create_chain(self): 
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
            {context}
            Question: {question}
            Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        return RetrievalQA.from_chain_type(
            self.llm,
            retriever = self.retriever,
            return_source_documents = True, 
            chain_type_kwargs = { "prompt": QA_CHAIN_PROMPT}
        )

    def create_retriever(self):
        # TODO: maybe change this line to use llama as the llm 
        llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct") # LLM in charge of compressing the document, can be changed to ollama
        compressor = LLMChainExtractor.from_llm(llm) 

        # contextual compression retriever -> in charge of retrieving the most relevant chunks and compressing them
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vecdb.as_retriever(search_type= "mmr") # max marginal relevance enforces diversity among results
        )

    def create_db(self, chunks, embeddings, db_dir):
        # 1. delete the contents of chroma dir if there are any
        os.system('rm -rf ./' + db_dir)
        vecdb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=(db_dir + '/')
        )
        return vecdb

    def split_pages(self, pages):
        # Split pages into chunks (paragraphs) for better semantic encoding
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450, # max number of characters in a chunk
            chunk_overlap=50, # number of characters to overlap between chunks
            separators=["\n\n", "\n", " ", ""]
        )
        return r_splitter.split_documents(pages)

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

