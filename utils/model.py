import torch
import torch.nn as nn

    
class RNN(nn.Module):

    def __init__(self,embedsize,vocabsize,outputsize):
        super(RNN,self).__init__()
        self.embd=nn.Embedding(vocabsize,embedsize) #5,embedsize
        self.rnn=nn.GRU(embedsize,embedsize,2)
        self.l1=nn.Linear(embedsize,outputsize)
      

    def forward(self,x):
        
        x=self.embd(x)
        x,_=self.rnn(x)
        x=self.l1(x[-1,:])
        return x
    