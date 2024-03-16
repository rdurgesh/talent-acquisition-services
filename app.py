from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/")
def home():
    return "<center><H1>Hello World from Flask! I am in Azure Cloud Now!</H1></center>"

@app.route('/parse_all_resumes', methods=['POST'])
def parse_all_resumes():
    return jsonify({'parsing_status': "success"})

@app.route('/search_matching_resumes', methods=['POST'])
def search_matching_resumes():
    return jsonify({'success': "200"})

@app.route('/parse_all_job_descriptions', methods=['POST'])
def parse_all_job_descriptions():
    return jsonify({'parsing_status': "success"})

@app.route('/search_matching_jobs', methods=['POST'])
def search_matching_jobs():
    return jsonify({'success': "200"})

if(__name__ == "main"):
    app.run(debug=True)