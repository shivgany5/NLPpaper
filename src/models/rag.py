import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RAGPipeline:
    """
    Retrieval-Augmented Generation approach using 8B model, ChromaDB, and a Re-ranker.
    """
    def __init__(self, config):
        model_id = config['models']['rag']['model_id']
        reranker_id = config['models']['rag']['re_ranker']
        self.top_k = config['models']['rag']['top_k']
        
        # Generator Setup
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            load_in_4bit=True, 
            device_map="auto"
        )
        
        # Retriever Setup
        self.embeddings = HuggingFaceEmbeddings(model_name=config['data']['embedding_model'])
        self.vectorstore = Chroma(
            persist_directory=config['data']['vector_db_path'], 
            embedding_function=self.embeddings
        )
        
        # Re-ranker Setup
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_id)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(reranker_id).eval()

    def retrieve_and_rerank(self, query: str) -> str:
        # Initial fast retrieval (2x the target K for reranking flexibility)
        docs = self.vectorstore.similarity_search(query, k=self.top_k * 2)
        
        # If no docs found, return empty context
        if not docs:
            return ""
            
        # Re-ranking using Cross-Encoder
        pairs = [[query, doc.page_content] for doc in docs]
        inputs = self.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            scores = self.reranker(**inputs).logits.squeeze(-1)
        
        # Sort by scores
        sorted_indices = torch.argsort(scores, descending=True)[:self.top_k]
        best_docs = [docs[i] for i in sorted_indices]
        
        # Combine sorted document chunk contexts
        return "\n\n".join([d.page_content for d in best_docs])

    def generate(self, query: str) -> str:
        context = self.retrieve_and_rerank(query)
        prompt = f"Using the following context extracted from classical literature, answer the query accurately.\n\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return generated_text.strip()
