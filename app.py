from flask import Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Hello World from Flask! I am in Azure Cloud Now!"

if(__name__ == "main"):
    app.run