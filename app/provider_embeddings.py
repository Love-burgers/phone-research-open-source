"""Count each bounded remote embedding batch against persistent provider limits."""
from langchain_core.embeddings import Embeddings
from .usage_guard import provider_quota
from .execution_context import remaining_seconds


class QuotaEmbeddings(Embeddings):
    def __init__(self,delegate,batch_size=64):
        self.delegate,self.batch_size=delegate,batch_size

    def embed_documents(self,texts):
        vectors=[]
        for start in range(0,len(texts),self.batch_size):
            batch=texts[start:start+self.batch_size]
            remaining_seconds()
            with provider_quota([{'embedding_inputs':batch}],0):
                vectors.extend(self.delegate.embed_documents(batch))
        return vectors

    def embed_query(self,text):
        remaining_seconds()
        with provider_quota([{'embedding_input':text}],0):
            return self.delegate.embed_query(text)
