from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pyresparser import ResumeParser
import os
from docx import Document
import uuid
import concurrent.futures
from pathlib import Path
from google.cloud import storage
import fitz  # PyMuPDF
from itertools import islice
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route("/")
def home():
    return "<center><H1>App Deployment Successful</H1></center>"

@app.route('/parse_all_resumes', methods=['POST'])
def parse_all_resumes():
    input_path = request.json['inputPath']
    bucket_name = input_path.split("/")[-1]

    path_to_private_key = 'fifth-compass-415612-76f634511b19.json'
    client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)
    bucket = client.bucket(bucket_name)

    str_folder_name_on_gcs = 'RESUME/data/'

    # Create the directory locally
    Path(str_folder_name_on_gcs).mkdir(parents=True, exist_ok=True)

    blobs = bucket.list_blobs(prefix=str_folder_name_on_gcs)

    # Limit to the first 100 blobs
    # TODO we should be removing this slicing of blobs (to avoid performance issues) before checking in
    limited_blobs = islice(blobs, 100)

    # Use concurrent.futures for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data_dicts = list(executor.map(process_blob, limited_blobs))

    # Write the data dictionary to an Excel file
    df = pd.DataFrame(data_dicts)
    excel_file = bucket_name + '_resume_data.xlsx'
    df.to_excel(excel_file, index=False)

    return jsonify({'status': "success"})

@app.route('/search_matching_resumes', methods=['POST'])
def search_matching_resumes():
    job_description = request.json['context']
    input_path = request.json['inputPath']
    category = request.json['category']
    threshold = request.json['threshold']
    no_of_matches = request.json['noOfMatches']
    bucket_name = input_path.split("/")[-1]

    # program to read the extracted data and process
    excel_file = bucket_name + '_resume_data.xlsx'
    df = pd.read_excel(excel_file)
    df = df.dropna()

    # stop word removal
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['skills'])

    # Apply K-means clustering on the resumes - currently on sample texts
    num_clusters = 30
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # TF-IDF vectorization for the input text
    input_vector = vectorizer.transform([job_description])

    # Calculate cosine similarity with each document in the cluster
    df['Similarity'] = df['skills'].apply(lambda x: cosine_similarity(vectorizer.transform([x]), input_vector).item())

    cluster_matches = df[df['Cluster'] == df.loc[df['Similarity'].idxmax(), 'Cluster']]
    cluster_matches = cluster_matches[cluster_matches['Similarity'] > 0]
    cluster_matches = cluster_matches.sort_values(by='Similarity', ascending=False)
    cluster_matches = cluster_matches.head(no_of_matches)

    # Prepare JSON response
    response_data = {
        "status": "success",
        "count": len(cluster_matches),
        "metadata": {
            "confidenceScore": cluster_matches['Similarity'].max() if not cluster_matches.empty else 0
        },
        "results": []
    }

    # Populate results in the JSON response
    for index, row in cluster_matches.iterrows():
        result = {
            "id": index + 1,
            "score": row['Similarity'],
            "path": row['name'] 
        }
        response_data['results'].append(result)

    return jsonify(response_data)


@app.route('/parse_all_job_descriptions', methods=['POST'])
def parse_all_job_descriptions():
    return jsonify({'status': "success"})

@app.route('/search_matching_jobs', methods=['POST'])
def search_matching_jobs():
    return jsonify({'success': "200"})

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_bytes):
    text = ""
    with fitz.open("pdf", pdf_bytes) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Function to parse resume using Pyresparser
def parse_resume(text):
    temp_file_path = str(uuid.uuid4()) + '.docx'
    doc = Document()
    doc.add_paragraph(text)
    doc.save(temp_file_path)

    try:
        data = ResumeParser(temp_file_path).get_extracted_data()
        return data
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

# Function to process a single blob (resume)
def process_blob(blob):
    resume_bytes = blob.download_as_bytes()
    text = extract_text_from_pdf(resume_bytes)
    data = parse_resume(text)
    skills = ', '.join(data.get('skills', []))
    data_dict = {
        'name': blob.name,
        'skills': skills
    }
    return data_dict

if __name__ == '__main__':
    app.run(debug=True)