# bring lightweight libraries
from flask import Flask, request, jsonify
import subprocess


# make an api
app = Flask(__name__)

# define a function to be called when the api is called
@app.route('/', methods=['GET'])
def predict():
   subprocess.call(['python3', 'vidFeed.py'])


# run the app
if __name__ == '__main__':
    app.run(debug=True)

