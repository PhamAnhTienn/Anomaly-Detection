import requests
import os
import torch
from utils.anomaly_detection import calculate_feature_anomaly_scores
from dotenv import load_dotenv
from groq import Groq

def search_wikipedia(feature_scores):
    search_query = "Energy consumption anomaly explanation. High scores in features:\n"
    for feature_name, score in feature_scores.items():
        search_query += f"{feature_name}: {score}\n"

    wiki_search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search_query,
        "format": "json",
        "srlimit": 3  
    }

    response = requests.get(wiki_search_url, params=params)
    
    if response.status_code == 200:
        search_results = response.json().get("query", {}).get("search", [])
        relevant_docs = [result['snippet'] for result in search_results]
    else:
        relevant_docs = []

    return relevant_docs

def generate_explanation_groq(feature_scores, relevant_docs):
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    client = Groq(api_key=api_key)

    prompt = (
        "You are an AI system explaining anomaly detection in energy consumption. "
        "The anomaly scores for different features are:\n"
    )
    for feature_name, score in feature_scores.items():
        prompt += f"{feature_name}: {score}\n"
    
    prompt += "\nBased on the following information from Wikipedia, explain possible reasons for this anomaly:\n"
    for doc in relevant_docs:
        prompt += f"Document: {doc}\n"
    
    prompt += "Provide a clear and concise explanation for this anomaly."

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps explain anomaly detection in energy consumption."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    explanation_text = chat_completion.choices[0].message.content
    return explanation_text

def explain_anomalies_for_groups(vae, lstm, test_windows, grouped_anomalies, threshold, feature_names):
    all_explanations = ""
    
    for idx, group in enumerate(grouped_anomalies, start=1):
        group_windows = [test_windows[i] for i in group]
        avg_window = torch.mean(torch.stack(group_windows), dim=0)

        feature_scores = calculate_feature_anomaly_scores(vae, lstm, avg_window, threshold, feature_names)

        relevant_docs = search_wikipedia(feature_scores)

        explanation_text = generate_explanation_groq(feature_scores, relevant_docs)

        all_explanations += "\n" + "=" * 80 + "\n"
        all_explanations += f"** Explanation for Anomaly Group {idx} (Indices: {group}) **\n"
        all_explanations += "=" * 80 + "\n"
        all_explanations += explanation_text + "\n"
        all_explanations += "=" * 80 + "\n\n"
    
    return all_explanations