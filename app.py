from flask import Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "<center><H1>Hello World from Flask! I am in Azure Cloud Now!</H1></center>"

if(__name__ == "main"):
    app.run