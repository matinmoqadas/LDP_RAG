import os
from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI

# AvalAI credentials
API_KEY = ""
BASE_URL = "https://api.avalai.ir/v1"

# Initialize AvalAI LLM
def get_eval_llm():
    return LangchainLLMWrapper(
        ChatOpenAI(
            base_url=BASE_URL,
            model="gpt-4o-mini",
            api_key=API_KEY
        )
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def create_embedding(text):
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    return resp.data[0].embedding

# Wrap the custom embedder for RAGAS
class CustomEmbeddingModel:
    def embed_query(self, text: str):
        return create_embedding(text)
    def embed_documents(self, texts: list[str]):
        return [create_embedding(t) for t in texts]

def get_eval_emb():
    return LangchainEmbeddingsWrapper(CustomEmbeddingModel())

def evaluate_ragas(user_input, retrieved_contexts, generated_answer, reference_answer):
    ds = Dataset.from_dict({
        "user_input": [user_input],
        "retrieved_contexts": [retrieved_contexts],
        "response": [generated_answer],
        "reference": [reference_answer],
    })

    llm = get_eval_llm()
    emb = get_eval_emb()

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        LLMContextPrecisionWithoutReference(llm=llm)
    ]

    result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=emb, show_progress=False)
    return result.to_pandas().iloc[0].to_dict()
