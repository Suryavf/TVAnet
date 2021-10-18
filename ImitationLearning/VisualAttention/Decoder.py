import torch
import torch.nn as nn
from IPython.core.debugger import set_trace

import ImitationLearning.VisualAttention.network.Gate as G

class CatDecoder(nn.Module):
    """ Constructor """
    def __init__(self,  HighEncoderNet, SpatialNet, FeatureNet,# CommandNet, 
                        LowLevelDim=128, HighLevelDim=512, 
                        n_hidden=1024, n_state=64,n_task=3,
                        study=False):
        super(CatDecoder, self).__init__()
        self.study = study

        # Parameters
        self.H =     n_hidden       # output LSTM   1024   2048
        self.R = int(n_hidden/4)    #  input LSTM    256    512
        self.S =     n_state
        self.n_task       =  n_task
        self.sequence_len =      20
        
        # Attention
        self.HighEncoder = HighEncoderNet
        self.SpatialAttn =     SpatialNet
        self.FeatureAttn =     FeatureNet
        # self. CmdDecoder =     CommandNet

        self.Gate = G.GRUGate(LowLevelDim)
        
        # Output
        self.dimReduction = nn.Conv2d(HighLevelDim,self.R, kernel_size=1, bias=False)
        self.lstm = nn.LSTM(  input_size = self.R,
                             hidden_size = self.H,
                              num_layers =      1)
        self.init_Wh   = nn.Linear(LowLevelDim,self.H,bias=True )
        self.init_Wc   = nn.Linear(LowLevelDim,self.H,bias=True )
        self.init_tanh = nn.Tanh()

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.normSpa  = nn.BatchNorm2d(LowLevelDim)
        self.ReLU     = nn.ReLU()  

        # Initialization
        torch.nn.init.xavier_uniform_(self.dimReduction.weight)
        torch.nn.init.xavier_uniform_(self.     init_Wh.weight)
        torch.nn.init.xavier_uniform_(self.     init_Wc.weight)
        self.lstm.reset_parameters()
        
        
    """ Initialize LSTM: hidden state and cell state
        Ref: Xu, Kelvin, et al. "Show, attend and tell: Neural 
            image caption generation with visual attention." 
            International conference on machine learning. 2015.

            * Input:  feature [batch,D,h,w]
            * Output: hidden  [1,batch,H]
                      cell    [1,batch,H]
    """
    def initializeLSTM(self,feature):
        with torch.no_grad():
            # Mean features
            feature = torch.mean(feature,(2,3)) # [batch,D,h,w] -> [batch,D]
            
            hidden = self.init_Wh(feature)  # [batch,D]*[D,H] -> [batch,H]
            hidden = self.init_tanh(hidden) # [batch,H]
            hidden = hidden.unsqueeze(0)    # [1,batch,H]

            cell = self.init_Wc(feature)    # [batch,D]*[D,H] -> [batch,H]
            cell = self.init_tanh(cell)     # [batch,H]
            cell = cell.unsqueeze(0)        # [1,batch,H]

            # (h,c) ~ [num_layers, batch, hidden_size]
            return hidden.contiguous(),cell.contiguous()
        
    """ Forward 
          - eta [batch,channel,high,width]
    """
    def forward(self,feature,cmd):
        # Parameters
        sequence_len = self.sequence_len
        if self.training: batch_size = int(feature.shape[0]/sequence_len)
        else            : batch_size =     feature.shape[0]
        _,C,H,W = feature.shape # Batch of Tensor Images is a tensor of (B, C, H, W) shape
        
        # Data
        if self.training: sequence = feature.view(batch_size,sequence_len,C,H,W).transpose(0,1) # [sequence,batch, ...]
        else            : sequence = feature  # [batch, ...]
        
        # Inicialize hidden state and cell state
        #   * hidden ~ [1,batch,H]
        #   * cell   ~ [1,batch,H]
        if self.training: xt = sequence[0]  # Input to inicialize LSTM 
        else            : xt = sequence[0].unsqueeze(0)
        ht,ct = self.initializeLSTM(xt)

        # Command decoder
        # cmd = self.CmdDecoder(command)
        if self.training: cmd = cmd.view(batch_size,sequence_len,-1).transpose(0,1) # [sequence,batch,4]

        # Prediction container
        st_,ht_ = list(),list()

        # State initialization
        if self.training: st = torch.cuda.FloatTensor(batch_size,self.n_task,self.S).uniform_()
        else            : st = torch.cuda.FloatTensor(         1,self.n_task,self.S).uniform_()
        
        # Study
        if self.study: α,β,F = list(),list(),list()
        else         : α,β,F =   None,  None,  None

        # Sequence loop
        n_range  = self.sequence_len if self.training else batch_size
        for k in range(n_range):
            # One time
            if self.training: ηt = sequence[k]              # [batch,L,D]
            else            : ηt = sequence[k].unsqueeze(0) # [  1  ,L,D]
            if self.training: cm = cmd[k]                   # [batch, 4 ]
            else            : cm = cmd[k].unsqueeze(0)      # [  1  , 4 ]

            # Spatial Attention
            xt, αt = self.SpatialAttn(ηt,st)
            xt = self.Gate(ηt,xt)
            
            # High-level encoder
            zt = self.HighEncoder(xt)

            # Feature-based attention
            # s[t] = f(z[t],h[t-1])
            _zt = self.avgpool1( zt)
            _zt = torch.flatten(_zt, 1)
            st, βt, Ft = self.FeatureAttn(_zt,ht[0],cm) # [batch,S]
            
            # Dimension reduction to LSTM
            rt = self.dimReduction(zt)
            rt = self.    avgpool2(rt)
            rt = torch.flatten(rt , 1)
            rt = rt.unsqueeze(0)
            
            # LSTM
            #  * yt     ~ [sequence,batch,H]
            #  * hidden ~ [ layers ,batch,H]
            #  * cell   ~ [ layers ,batch,H]
            _,(ht,ct)= self.lstm(rt,(ht,ct))
            
            # Output
            st_.append(st.squeeze())    # [batch,n_task,S]
            ht_.append(ht.squeeze())    # [batch,       H]

            # Study
            if self.study:
                α.append(αt)
                β.append(βt)
                F.append(Ft)

        # Concatenate
        st_ = torch.stack(st_,dim=0)
        ht_ = torch.stack(ht_,dim=0)
        if self.training: 
            st_ = st_.transpose(0,1).reshape(batch_size*sequence_len,self.n_task,self.S)
            ht_ = ht_.transpose(0,1).reshape(batch_size*sequence_len,            self.H)

        # Compile study
        if self.study:
            α = torch.stack(α, dim=0)
            β = torch.stack(β, dim=0)
            F = torch.stack(F, dim=0)
        
        return st_, {'hidden': ht_, 'feature': F}, {'alpha': α, 'beta': β}
        
