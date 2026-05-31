import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
print("TruthLens project started")
print("Pandas imported successfully")
question = "How can SQL injection attacks be prevented?"
print(question)
responses = [
    "SQL injection can be prevented by using prepared statements and validating user input.",
    "Using parameterized queries and proper input sanitization helps avoid SQL injection attacks.",
    "Input validation, least privilege access, and ORM frameworks reduce SQL injection risks."
]

print(responses)
data = {
    "question_id": ["Q1", "Q1", "Q1"],
    "question": [question, question, question],
    "response_id": ["R1", "R2", "R3"],
    "response_text": responses
}

df = pd.DataFrame(data)
print(df)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["response_text"])

print("TF-IDF matrix shape:", tfidf_matrix.shape)
similarity_matrix = cosine_similarity(tfidf_matrix)
print(similarity_matrix)
print("Cosine similarity matrix calculated successfully")
import numpy as np

# Ignore diagonal (self-similarity)
n = similarity_matrix.shape[0]
consistency_score = (
    similarity_matrix.sum() - n
) / (n * (n - 1))

print("Consistency Score:", round(consistency_score, 3))
original_question = "How can SQL injection attacks be prevented?"

rephrased_questions = [
    "What methods can be used to stop SQL injection?",
    "How do developers protect applications from SQL injection attacks?",
    "Ways to defend against SQL injection vulnerabilities?"
]

print("Original Question:", original_question)
print("Rephrased Questions:", rephrased_questions)
rephrased_responses = [
    "Developers can prevent SQL injection using parameterized queries and input validation.",
    "Protecting applications from SQL injection involves sanitizing inputs and using prepared statements.",
    "Defending against SQL injection requires secure coding practices like validation and least privilege."
]
all_responses = df["response_text"].tolist() + rephrased_responses

tfidf_all = vectorizer.fit_transform(all_responses)
similarity_all = cosine_similarity(tfidf_all)
original_vectors = tfidf_all[:len(df)]
rephrased_vectors = tfidf_all[len(df):]

stability_scores = cosine_similarity(original_vectors, rephrased_vectors)
stability_score = stability_scores.mean()

print("Stability Score:", round(stability_score, 3))
expected_keywords = [
    "input validation",
    "prepared statements",
    "parameterized queries",
    "least privilege",
    "secure coding",
    "orm"
]
def compute_coverage(responses, keywords):
    coverage_count = 0
    for keyword in keywords:
        for response in responses:
            if keyword.lower() in response.lower():
                coverage_count += 1
                break
    return coverage_count / len(keywords)

coverage_score = compute_coverage(df["response_text"].tolist(), expected_keywords)
print("Coverage Score:", round(coverage_score, 3))
# Weights (can be justified in paper)
w_consistency = 0.4
w_stability = 0.3
w_coverage = 0.3

final_trust_score = (
    w_consistency * consistency_score +
    w_stability * stability_score +
    w_coverage * coverage_score
)

print("FINAL TRUST SCORE:", round(final_trust_score, 3))
# -------------------------
# 6. Professional Graph with Gradient Colors
# -------------------------
metrics = ["Consistency", "Stability", "Coverage", "Final Trust"]
scores = [consistency_score, stability_score, coverage_score, final_trust_score]

sns.set(style="whitegrid")
plt.figure(figsize=(9,6))

# Create gradient colors
colors = sns.color_palette("Blues_d", n_colors=len(scores))

# Plot bars with gradient colors
for i, (metric, score) in enumerate(zip(metrics, scores)):
    plt.bar(metric, score, color=colors[i])
    plt.text(i, score + 0.02, f"{score:.2f}", ha='center', fontweight='bold', fontsize=11)

plt.title("Trust Metrics of TruthLens Framework", fontsize=16, fontweight='bold')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Metric", fontsize=12)
plt.ylim(0,1.0)

sns.despine()
plt.savefig("TruthLens_TrustMetrics.png", dpi=300, bbox_inches='tight')
plt.show()
