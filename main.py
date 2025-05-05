import openai
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json

with open('data.json', 'r') as file:
    data = json.load(file)

try:
    from api_key import SECRET_KEY
    openai.api_key = SECRET_KEY
except FileNotFoundError:
    openai.api_key = 'API_ANAHTARINIZI_BURAYA_GİRİN'

client = OpenAI()
class AiAgent:
    def __init__(self):
        self.qa = data
        self.question_embedding = {}
        self.generate_embedding()

    def generate_embedding(self):
        if os.path.exists('embeddings_cache.pkl'):
            with open('embeddings_cache.pkl', 'rb') as f:
                self.question_embedding = pickle.load(f)
                return

        for question in self.qa.keys():
            self.question_embedding[question] = self.get_embedding(question)

        with open('embeddings_cache.pkl', 'wb') as f:
            pickle.dump(self.question_embedding, f)

    def get_embedding(self, text):
        response = client.embeddings.create(
            model= "text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def find_closest_question(self, query, similarity_threshold=0.7):
        query_embedding = self.get_embedding(query)

        best_match = None
        best_similarity = 0

        for question, embedding in self.question_embedding.items():
            similarity = cosine_similarity(
                [query_embedding],
                [embedding]
            )[0][0]

            if similarity > best_similarity:
                best_match = question
                best_similarity = similarity

        if best_similarity > similarity_threshold:
            return best_match, best_similarity
        return None, best_similarity

    def answer_question(self, query):
        closest_question, similarity = self.find_closest_question(query)

        if closest_question:
            return {
                'answer': self.qa[closest_question],
                'matched_question': closest_question,
                'similarity': similarity,
            }
        else:
            return {
                'answer': "Üzgünüm, bu konu hakkında şuanki bilgilerimle cevap veremiyorum.",
                'matched_question': None,
                'similarity': similarity
            }


if __name__ == "__main__":
    agent = AiAgent()

    while True:
        user_query = input("\nSoru Sor (Çıkmak için 'exit' yaz): ")
        if user_query.lower() == 'exit':
            break

        result = agent.answer_question(user_query)
        print(f"\nCevap: {result['answer']}")

        if result['matched_question']:
            print(f"Eşleşen soru: '{result['matched_question']}'")
            print(f"Benzerlik skoru: {result['similarity']:.2f}")