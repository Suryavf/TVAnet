import math 
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttnNet(nn.Module):
    """ Constructor """
    def __init__(self, cube_size,n_head,n_state,study=False):
        super(SpatialAttnNet, self).__init__()
        # Parameters 
        self.high  = cube_size[0]
        self.width = cube_size[1]
        self.D = cube_size[2]           #  depth
        self.L = self.high*self.width   #  h x w
        self.R = self.L*self.D          #  L x D
        self.study = study

        # Deberian ser entrada
        self.S = n_state

        self.h = n_head
        self.d = int(self.D/self.h)
        self.hd = self.d*self.h
        self.sqrtd = self.d ** .5

        # Spatial 
        self.to_q = nn.Conv2d(self.D, self.hd, 1, bias = False)
        self.to_v = nn.Conv2d(self.D, self.hd, 1, bias = False)
        self.to_k = nn.Linear(self.S, self.hd,    bias = False)

        self.fc = nn.Linear(self.hd, self.D)

        # Initialization
        torch.nn.init.xavier_uniform_(self.to_q.weight)
        torch.nn.init.xavier_uniform_(self.to_k.weight)
        torch.nn.init.xavier_uniform_(self.to_v.weight)
        torch.nn.init.xavier_uniform_(self. fc .weight)

        self.normSpa = nn.GroupNorm( 1,self.D )
        self.normFtr = nn.LayerNorm(   self.S )

        self.Tanh    = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
        self.ReLu    = nn.ReLU()
        self.Softmax = nn.Softmax(2)
    
    def norm(self,x,p=1):
        x = self.Sigmoid(x)
        y = torch.norm(x,p=p,dim=2,keepdim=True)
        return x/y
        

    """ Forward 
          - eta [batch,channel,high,width]
          -  F  [batch,n,M]
    """
    def forward(self,ηt,Ft0):
        # Batch norm
        # L: high x weight
        # d: visual depth
        # h: numero of heads
        # n: number of tasks
        ηt  = self.normSpa(ηt )
        Ft0 = self.normFtr(Ft0)

        # Query, key, value
        Q = self.to_q(ηt )     # [batch,hd,L]
        K = self.to_k(Ft0)     # [batch,n,hd]
        V = self.to_v(ηt )     # [batch,hd,L]

        K = K.transpose(1,2)    # [batch,hd,n]

        Q = Q.view(-1, self.hd, self.L).transpose(1,2) # [batch,hd,L]
        V = V.view(-1, self.hd, self.L).transpose(1,2) # [batch,hd,L]

        # Attention map
        Q,K,V = map(lambda x: x.reshape(x.shape[0],self.h,self.d,-1),[Q,K,V])   # Q,V -> [batch,h,d,L]
                                                                                #  K  -> [batch,h,d,n]
        QK = torch.einsum('bhdn,bhdm->bhnm', (Q,K))     # [batch, h, L, n]
        A  = self.Softmax(QK/self.sqrtd) # norm4(QK)    # [batch, h, L, n]

        # Apply
        Z  = torch.einsum('bhnk,bhdn->bnhd', (A,V))     # [batch, L, h, d]
        
        # Output
        Z = Z.view(-1,self.L,self.hd)
        Z = self.fc(Z)          # [batch, L, D]
        Z = Z.transpose(1,2)    # [batch, D, L]
        Z = Z.reshape(-1,self.D,self.high,self.width).contiguous()  # [batch, D, high, width]
        
        if self.study: return Z, A.squeeze(3)
        else         : return Z, None
        

""" Feature attention network
    -------------------------
        * Input 
            - n_encode: depth of visual feature input
            - n_hidden: size of hidden state of LSTM
            - n_command: size of command encoding
            - n_state: output dimension (state)
"""
class FeatureAttnNet(nn.Module):
    """ Constructor """
    def __init__(self, n_encode, n_hidden, n_command, n_state, n_feature, n_task, study=False):
        super(FeatureAttnNet, self).__init__()
        self.n_feature = n_feature
        self.D         = n_state
        self.sqrtDepth = math.sqrt(self.D)
        self.study     = study
        self.magF      = False

        self.h = n_task  # Multi-task
        self.M = self.D*int(self.n_feature/2)

        # Feature 
        self.wz = nn.Linear( n_encode, self.M, bias = False)
        self.wh = nn.Linear( n_hidden, self.M, bias = False)

        self.to_q = nn.Linear(n_command, self.D*self.h, bias = False)
        self.to_k = nn.Linear(  self.D , self.D*self.h, bias = False)
        self.to_v = nn.Linear(  self.D , self.D*self.h, bias = False)
        
        self.Lnorm   = nn.LayerNorm( n_state )
        self.Softmax = nn.Softmax(2)
        
        # Initialization
        torch.nn.init.xavier_uniform_(self. wz .weight)
        torch.nn.init.xavier_uniform_(self. wh .weight)
        torch.nn.init.xavier_uniform_(self.to_q.weight)
        torch.nn.init.xavier_uniform_(self.to_k.weight)
        torch.nn.init.xavier_uniform_(self.to_v.weight)
    

    """ Forward 
          - feature [batch,  channel]
          - hidden  [batch,n_hidden ]
          - command [batch,n_command]
    """
    def forward(self,feature,hidden,command):
        batch = feature.shape[0]
        # h: number of tasks
        # d: size of state (depth)
        # n: number of features Fn

        z = F.gelu(self.wz(feature)) # [batch,dn/2]
        h = F.gelu(self.wh( hidden)) # [batch,dn/2]
        y = torch.cat([z,h],dim=1)      # [batch,dn]
        y = y.reshape(batch,-1,self.D)  # [batch,n,d]
        y = self.Lnorm(y)               # [batch,n,d]
        
        # Query, key, value
        Q = self.to_q(command)              # [batch,hd]
        K = self.to_k(y).transpose(2,1)     # [batch,n,hd] -> [batch,hd,n]
        V = self.to_v(y).transpose(2,1)     # [batch,n,hd] -> [batch,hd,n]

        Q,K,V = map(lambda x: x.reshape(batch,self.h,self.D,-1),[Q,K,V])    # K,V -> [batch,h,d,n]
                                                                            #  Q  -> [batch,h,d,1]
        
        # Attention 
        QK = torch.einsum('bhdn,bhdm->bhmn', (Q,K))     # [batch,h,n,1]
        if self.magF:
            mV = torch.norm(y,p=1,dim=2)/self.D         # [batch,n]
            mV = mV.unsqueeze(2).unsqueeze(1)           # [batch,1,n,1]
        else:
            mV = torch.norm(V,p=1,dim=2)/self.D         # [batch,h,n]
            mV = mV.unsqueeze(3)                        # [batch,h,n,1]
        A  = self.Softmax(mV*QK/self.sqrtDepth)         # [batch,h,n,1]

        # Apply
        S = torch.einsum('bhnm,bhdn->bhdm', (A,V))      # [batch,h,d,1]
        S = S.view(batch,self.h,-1)                     # [batch,h,d]

        if self.study: return S,   A,   y
        else         : return S,None,None


class CommandNet(nn.Module):
    """ Constructor """
    def __init__(self,n_encode=16):
        super(CommandNet, self).__init__()
        self.Wc   = nn.Linear( 4, n_encode, bias= True)
        self.ReLU = nn.ReLU()
        # Initialization
        nn.init.xavier_uniform_(self.Wc.weight)

    def forward(self,control):
        c = control*2-1
        c = self.Wc(c)
        return self.ReLU(c)
        
