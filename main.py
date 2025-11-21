import json
from main_MedRAG import get_query_embedding, Faiss, documents, document_embeddings, generate_diagnosis_report
from authentication import augmented_features_path

query = """
A 45-year-old female with chronic low back pain radiating to left leg, worse when sitting, tingling sensation.
"""

query_embedding = get_query_embedding(query)

topk = 3
indices = Faiss(document_embeddings, query_embedding, k=topk)
retrieved_documents = [documents[i] for i in indices[0]]
print("retrieved_documents:",retrieved_documents)

# final_retrieved_info = []
# for doc_path in retrieved_documents:
#     with open(doc_path, 'r') as f:
#         patient_case = json.load(f)
    
#     final_retrieved_info.append(patient_case)

# report = generate_diagnosis_report(
#     augmented_features_path,
#     query,
#     final_retrieved_info,
#     top_n=1,
#     match_n=5
# )

# print("=== RETRIEVED DOCS ===")
# print(retrieved_documents)

# print("\n=== GENERATED DIAGNOSIS REPORT ===")
# print(report)
