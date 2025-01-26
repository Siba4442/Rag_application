import os
import chromadb
from typing import List
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

class DocLoader:
    def __init__ (self,
                  doc_path : str,
                  client_type : str,
                  verctordb_path : str,
                  collection_name : str,
                  sep : str,
                  chunk_size : int,
                  chunk_overlap : int,
                  ef) -> None:
        
        
        self.doc_path =  doc_path
        self.client_type =  client_type
        self.verctordb_path =  verctordb_path
        self.collection_name =  collection_name
        self.separator =  sep
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap
        self.embedding_function = ef
        
    def get_client(self):
        
        if self.client_type == 'chroma':
            return PersistentClient(path=self.verctordb_path)
        
    def get_text_splitter(self,
                          sep: str,
                          chunk_size: int,
                          chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        
        txt_splitter = RecursiveCharacterTextSplitter(
                            chunk_size = self.chunk_size,
                            chunk_overlap = self.chunk_overlap,
                            length_function = len,
                            is_separator_regex = False
                        )
        return txt_splitter
    
    
    def get_langchain_embedding(self) -> List[List[float]]:
        return create_langchain_embedding(self.embedding_function)
    
    
    def doc_reader(self, flag = True, singlefile = '') -> list:
        
        documents = []
        
        def pdfloader(pdf_path, file) -> list:
            
            tmp_doc_list = []
            loader = PyPDFLoader(pdf_path)
            tmp_doc_list.extend(loader.load())
            
            for docid, doc in enumerate(tmp_doc_list):
                doc.metadata = {
                    'page': str(docid + 1),
                    'source': file
                }
            
            return tmp_doc_list
        
        
        if flag == False:
            for file in os.listdir(self.doc_path):
                if file.endswith('.pdf'):
                    pdf_path = self.doc_path + "/" + file
                    documents.extend(pdfloader(pdf_path, file))
                else:
                    print("There is no PDF file.... ")
                    
        else:
            if singlefile.endswith('.pdf'):
                pdf_path = self.doc_path + '/' + singlefile
                documents.extend(pdfloader(pdf_path, singlefile))
            else:
                print("There is no PDF file......")
                
        return documents
    
    def create_update_vectorstore(self, file = "") -> None:
        
        if file == "":
            flag = False
        else:
            flag = True
            
        text_splitter = self.get_text_splitter(self.separator,
                                               self.chunk_size,
                                               self.chunk_overlap)
        
        chunked_documents = text_splitter.split_documents(self.doc_reader(flag, file))
        
        Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.get_langchain_embedding(),
            collection_name=self.collection_name,
            client=self.get_client()
        )
        
        print(f"Added {len(chunked_documents)} chunks to chroma db")
        
    def get_vector_store(self):
        
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.verctordb_path
        )
        
        return vector_store
    
    
    
    def del_scorce_file(self, file) -> None:
        
        client=self.get_client()
        collection=client.get_collection(name=self.collection_name)
        
        metadata_list = collection.get()['metadatas']
        
        file_name = []
        
        for metadata in metadata_list:
            filename = metadata['source'].split('\\')[-1]
            if filename not in file_name:
                file_name.append(filename)
                
        print("Printing pdf names extracted from metadata before delecting one of them....")
        print(file_name)
        
        if file in file_name:
            collection.delete(where={"source": file})