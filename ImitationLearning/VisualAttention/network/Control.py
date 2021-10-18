import torch
import torch.nn as nn
import torch.nn.functional as F


""" Policy: Multi-task + DenseNet
    -----------------------------
    Ref: https://sites.google.com/view/d2rl/home
        * Input: feature [batch,L,D]
                 hidden  [1,batch,H]
        * Output: alpha  [batch,L,1]
"""
class DenseNet(nn.Module):
    def __init__(self, n_depth, n_out):
        super(DenseNet, self).__init__()

        self.n_input  = 2*n_depth
        self.n_hidden =   n_depth

        self.w1 = nn.Linear(self.n_hidden,self.n_hidden)
        self.w2 = nn.Linear(self.n_input ,self.n_hidden)
        self.w3 = nn.Linear(self.n_input ,self.n_hidden)
        self.w4 = nn.Linear(self.n_hidden,    n_out    )

        # Initialization
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.xavier_uniform_(self.w4.weight)
    
    def forward(self,x):
        z = F.relu(self.w1(x))
        z = torch.cat([z,x],dim=1)
        
        z = F.relu(self.w2(z))
        z = torch.cat([z,x],dim=1)
        
        z = F.relu(self.w3(z))
        return self.w4(z)
        

class MultiTaskPolicy(nn.Module):
    def __init__(self, n_depth,vel_manager=False):
        super(MultiTaskPolicy, self).__init__()
        self.manager = vel_manager

        # Nets
        self.steering     = DenseNet(n_depth,1)
        self.acceleration = DenseNet(n_depth,2)
        
        if self.manager:
            self.  switch = DenseNet(n_depth,3)
            self.LogSoftmax = nn.LogSoftmax(dim=1)
            self.   Softmax = nn.   Softmax(dim=1)

    """ Forward 
          - state [batch,n_task,depth]
    """
    def forward(self,state):
        # Execute
        st = self.    steering(state[:,0,:])
        at = self.acceleration(state[:,1,:])
        
        if self.manager:
            # Velocity manager
            dc = self.switch(state[:,2,:])
            manager = self.LogSoftmax(dc)
            mask    = self.   Softmax(dc)
            # Masked
            at = at*mask[:,1:]
        else:
            manager = None

        return torch.cat([st,at],dim=1),manager
        

class MultiTaskPolicy2(nn.Module):
    def __init__(self, n_depth,n_cmd,vel_manager=False):
        super(MultiTaskPolicy2, self).__init__()
        self.manager = vel_manager

        # Nets
        self.steering     = DenseNet(n_depth+n_cmd,1)
        self.acceleration = DenseNet(n_depth+n_cmd,2)
        self.ws   = nn.Linear(n_cmd,n_cmd)
        self.wa   = nn.Linear(n_cmd,n_cmd)
        self.ReLU = nn.ReLU()  
        
        if self.manager:
            self.  switch = DenseNet(n_depth+n_cmd,3)
            self.LogSoftmax = nn.LogSoftmax(dim=1)
            self.   Softmax = nn.   Softmax(dim=1)
            
        # Initialization
        torch.nn.init.xavier_uniform_(self.ws.weight)
        torch.nn.init.xavier_uniform_(self.wa.weight)
        

    """ Forward 
          - state [batch,n_task,depth]
    """
    def forward(self,state,ct):
        # Execute
        st = self.ReLU(self.ws(ct))
        st = torch.cat([state[:,0,:],st],dim=1)
        st = self.steering(st)

        ct = self.ReLU(self.wa(ct))
        at = torch.cat([state[:,1,:],ct],dim=1)
        at = self.acceleration(at)
        
        if self.manager:
            # Velocity manager
            dc = torch.cat([state[:,2,:],ct],dim=1)
            dc = self.switch(dc)
            manager = self.LogSoftmax(dc)
            mask    = self.   Softmax(dc)
            # Masked
            at = at*mask[:,1:]
        else:
            manager = None

        return torch.cat([st,at],dim=1),manager
        
