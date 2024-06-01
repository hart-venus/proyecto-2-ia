import os
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add parent directory to path variable

class RagApi: 
    def __init__(self, docs_dir='docs'):
        self.docs_dir = docs_dir
        docs = self.find_docs(docs_dir, '.pdf') # find all pdf files in docs dir
        print(docs)


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

