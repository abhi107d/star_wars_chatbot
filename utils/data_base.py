
import pandas as pd
from nltk.tokenize import word_tokenize
import torch
import random 
import string
class DataBase:

    def __init__(self,path):
        
        data=pd.read_json(path)
        self.data=data.to_dict()
        vocab=[]

        for i in self.data['intents'].values():
            for j in i['patterns']:
                for k in word_tokenize(j):
                    if k not in string.punctuation:
                        vocab.append(k.lower())      
             
        vocab=sorted(set(vocab))
        self.wordindx={}
        for i,j in enumerate(vocab):
            self.wordindx[j]=i

    def convertIndex(self,sentence):
        token=[i.lower() for i in word_tokenize(sentence)]
        vec=[]
        for i in token:
            if i in self.wordindx.keys():
                vec.append(self.wordindx[i])

        if len(vec)!=0:        
            return torch.tensor(vec)
        return None
    
    def generateResponse(self,index):
        return random.choice(self.data['intents'][index]['responses'])
        