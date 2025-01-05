import torch

import warnings
import pandas as pd
import  sys

sys.path.insert(1, './utils/')
from data_base import DataBase
from model import RNN



warnings.filterwarnings("ignore", category=FutureWarning)
device=torch.device('cuda' if torch.cuda.is_available() else 'gpu')
d=DataBase('./Data/starwarsintents.json')
model=torch.load("./model/starwars.pth").to(device).eval()


op=None
while(op!=1):
    user=input("User: ")
    vec=d.convertIndex(user)
    if vec is not None:
        op=model(vec.to(device)).argmax(0).item()
        print("sbot: ",d.generateResponse(op))
    else:
        print('Im a little dum idk what ure sayin')
        








    
