from flask import Flask,render_template
from flask import Flask, request, jsonify
import warnings
import sys
import torch
sys.path.insert(1, './utils/')
from data_base import DataBase
from model import RNN



warnings.filterwarnings("ignore", category=FutureWarning)
device=torch.device('cuda' if torch.cuda.is_available() else 'gpu')
d=DataBase('./Data/starwarsintents.json')
model=torch.load("./model/starwars.pth").to(device).eval()



app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')


@app.route('/response', methods=['POST'])
def send_response():
    data = request.get_json() 
    user_message = data.get('message')  

    vec=d.convertIndex(user_message)
    if vec is not None:
        op=model(vec.to(device)).argmax(0).item()
        bot_response = d.generateResponse(op)
    else:
        bot_response='Im a little dum idk what ure sayin'
    
  
    
    return jsonify({"response": bot_response})  

if __name__ == "__main__":
    app.run(debug=True)
