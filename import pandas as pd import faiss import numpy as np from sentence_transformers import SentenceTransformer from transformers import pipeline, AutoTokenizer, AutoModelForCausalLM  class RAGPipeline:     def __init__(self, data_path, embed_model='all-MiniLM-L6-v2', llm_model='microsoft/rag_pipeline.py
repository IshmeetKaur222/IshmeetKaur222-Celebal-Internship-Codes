
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class RAGPipeline:
    def __init__(self, data_path, embed_model='all-MiniLM-L6-v2', llm_model='microsoft/phi-2', top_k=3):
        self.data_path = data_path
        self.embed_model = SentenceTransformer(embed_model)
        self.llm_model = llm_model
        self.top_k = top_k
        self.texts = self._load_data()
        self.index = self._build_index()
        self.llm = self._load_llm()

    def _load_data(self):
        df = pd.read_csv(self.data_path).fillna("unknown")
        return df.apply(lambda row: " ".join([f"{col}: {row[col]}" for col in df.columns]), axis=1).tolist()

    def _build_index(self):
        embeddings = self.embed_model.encode(self.texts, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def _load_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        model = AutoModelForCausalLM.from_pretrained(self.llm_model)
        return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

    def retrieve(self, query):
        query_vec = self.embed_model.encode([query], convert_to_numpy=True)
        _, I = self.index.search(query_vec, self.top_k)
        return [self.texts[i] for i in I[0]]

    def generate_answer(self, query):
        context = "\n".join(self.retrieve(query))
        prompt = f"""You are an assistant helping with loan approval prediction.

Context:
{context}

Question: {query}
Answer:"""
        response = self.llm(prompt)[0]['generated_text']
        return response.split("Answer:")[-1].strip()
