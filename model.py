import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GDCN(nn.Module):
    def __init__(self, in_dim=1, blocks=2, layers=2, residual_channels=1, dilation_channels=1, kernel_size=2):
        super(GDCN, self).__init__()
        self.blocks = blocks
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1),
                                    padding='same')

        for b in range(blocks):
            new_dilation = 1
            for i in range(layers):
                # dialated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation, padding='same'))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation, padding='same'))

    def forward(self, input):
        x = torch.unsqueeze(torch.unsqueeze(input, 0), 0)
        x = self.start_conv(x)
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
        x = torch.squeeze(x, 1)
        x = torch.permute(x, (0, 2, 1))
        x = torch.squeeze(x)
        x = torch.nan_to_num(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_features=200, out_features=200, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).to(device))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)).to(device))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        # adj is the adjacency matrix
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = adj
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention.float(), Wh)
        return torch.nan_to_num(F.elu(h_prime))

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #         print(self.weight.data)
    #         print(self.bias.data)

    def forward(self, input):
        return F.linear(input, self.filter_square_matrix.matmul(self.weight.double()), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class GCN(nn.Module):
    def __init__(self, A, feature_size):
        super(GCN, self).__init__()
        self.feature_size = feature_size
        self.A = A
        self.linear = FilterLinear(410, 410, self.A, bias=False)

    def forward(self, input, BNP):
        # some squeeze and unsqueeze
        x = torch.einsum('ij,xyij -> xyij', self.A, torch.Tensor(BNP).double())
        x = self.linear(x)
        x = x.reshape(-1, 410, 410)
        x = torch.einsum('ij,ijk->ik', input.double().transpose(0, 1), x.double())
        return torch.nan_to_num(x.transpose(0, 1))


class TSM_first(nn.Module):
    def __init__(self, A, in_features=200, out_features=200):
        super(TSM_first, self).__init__()
        self.GDCN1 = GDCN()
        self.GDCN2 = GDCN()
        self.gat = GAT(in_features=in_features, out_features=out_features)
        self.A = A
        self.GCN1 = GCN(A, in_features)
        self.GCN2 = GCN(A, in_features)
        self.BN1 = nn.BatchNorm1d(in_features)
        self.BN2 = nn.BatchNorm1d(in_features)

        self.nodevec1 = nn.Parameter(torch.randn(int(A.shape[0]), 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, int(A.shape[0])).to(device), requires_grad=True).to(device)

        self.nodevec3 = nn.Parameter(torch.randn(int(A.shape[0]), 10).to(device), requires_grad=True).to(device)
        self.nodevec4 = nn.Parameter(torch.randn(10, int(A.shape[0])).to(device), requires_grad=True).to(device)

    def forward(self, input, BNP):
        BNP = torch.squeeze(BNP)
        x = input.reshape((input.shape[0] * input.shape[1]), input.shape[2])
        x1 = self.GDCN1(x)
        x_middle = self.gat(x1, self.A)
        x_middle = self.GCN1(x_middle, BNP)
        x = self.BN1(x.permute(1, 0) + x_middle.float()).permute(1, 0)

        x2 = self.GDCN2(x)
        x_middle = self.gat(x2, self.A)
        x_middle = self.GCN2(x_middle, BNP)
        x = self.BN2(x.permute(1, 0) + x_middle.float()).permute(1, 0)

        return x, x1, x2


class TSM_others(nn.Module):
    def __init__(self, A, in_features=200, out_features=200):
        super(TSM_others, self).__init__()
        self.GDCN1 = GDCN()
        self.GDCN2 = GDCN()
        self.gat = GAT(in_features=in_features, out_features=out_features)
        self.A = A
        self.GCN1 = GCN(A, in_features)
        self.GCN2 = GCN(A, in_features)
        self.BN1 = nn.BatchNorm1d(in_features)
        self.BN2 = nn.BatchNorm1d(in_features)

        self.nodevec1 = nn.Parameter(torch.randn(int(A.shape[0]), 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, int(A.shape[0])).to(device), requires_grad=True).to(device)

        self.nodevec3 = nn.Parameter(torch.randn(int(A.shape[0]), 10).to(device), requires_grad=True).to(device)
        self.nodevec4 = nn.Parameter(torch.randn(10, int(A.shape[0])).to(device), requires_grad=True).to(device)

    def forward(self, x, BNP):
        BNP = torch.squeeze(BNP)
        x1 = self.GDCN1(x)
        x_middle = self.gat(x1, self.A)
        x_middle = self.GCN1(x_middle, BNP)
        x = self.BN1(x.permute(1, 0) + x_middle.float()).permute(1, 0)

        x2 = self.GDCN2(x)
        x_middle = self.gat(x2, self.A)
        x_middle = self.GCN2(x_middle, BNP)
        x = self.BN2(x.permute(1, 0) + x_middle.float()).permute(1, 0)

        return x, x1, x2


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class COMPONENT(nn.Module):
    def __init__(self, A, inp=200, out=200):
        super(COMPONENT, self).__init__()
        self.A = A
        self.TSM1 = TSM_first(A=torch.tensor(self.A), in_features=inp, out_features=out)
        self.TSM2 = TSM_others(A=torch.tensor(self.A), in_features=inp, out_features=out)
        self.TSM3 = TSM_others(A=torch.tensor(self.A), in_features=inp, out_features=out)
        self.TSM4 = TSM_others(A=torch.tensor(self.A), in_features=inp, out_features=out)
        self.RL = nn.ReLU()
        self.conv1 = linear(1, 1)
        self.conv2 = linear(1, 1)
        self.linear = nn.Linear(inp, out_features=20)

    def forward(self, x, BNP):
        x, x11, x12 = self.TSM1(x, BNP)
        x, x21, x22 = self.TSM2(x, BNP)
        x, x31, x32 = self.TSM3(x, BNP)
        x, x41, x42 = self.TSM4(x, BNP)

        x = self.RL(x11 + x12 + x21 + x22 + x31 + x32 + x41 + x42)
        x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
        x = self.conv1(x)
        x = self.RL(x)
        x = self.conv2(x)
        x = self.RL(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        return x.permute(1, 0)


class Multi_STGAC(nn.Module):
    def __init__(self, A):
        super(Multi_STGAC, self).__init__()
        self.A = A
        self.Recent_component = COMPONENT(A, 20 * 10, 20 * 10)  # 20 is the batch size and 10 is the sequence length
        self.Daily_component = COMPONENT(A, 20 * 4, 20 * 4)
        self.Weekly_component = COMPONENT(A, 20 * 2, 20 * 2)
        self.conv1 = nn.Conv2d(in_channels=3 * 410, out_channels=410, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=410, out_channels=410, kernel_size=(1, 1), padding='same')
        self.elu = nn.ELU()

    def forward(self, Xr, Xd, Xw, BNPr, BNPd, BNPw):
        X1 = self.Recent_component(Xr, BNPr)
        X2 = self.Daily_component(Xd, BNPd)
        X3 = self.Weekly_component(Xw, BNPw)
        # Concatenation operation
        Y = torch.cat((X1, X2, X3), 1)
        Y = torch.unsqueeze(torch.unsqueeze(Y, 2), 3)
        Y = self.conv1(Y)
        Y = self.elu(Y)
        Y = self.conv2(Y)
        Y = torch.squeeze(Y)
        return Y


class ZINB(nn.Module):
    def __init__(self, n_input, n_hidden, p_dropout):
        super(ZINB, self).__init__()
        self.dense1 = nn.Linear(n_input, n_hidden)
        self.dense2 = nn.Linear(n_hidden, n_hidden)
        self.dropout = nn.Dropout(p_dropout)
        self.dense_n = nn.Linear(n_hidden, 1)
        self.dense_p = nn.Linear(n_hidden, 1)
        self.dense_pi = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        n = torch.exp(self.dense_n(x))
        p = torch.sigmoid(self.dense_p(x))
        pi = torch.sigmoid(self.dense_pi(x))
        return n, p, pi

    def zinb_loss(self, x, n, p, pi, eps=1e-8):
        nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=n, probs=p + eps)
        x = torch.clamp(torch.round(x), min=0)
        log_prob = nb.log_prob(x)
        log_l = torch.log(pi + (1 - pi) * torch.exp(log_prob))
        log_1ml = torch.log(1 - pi + pi * torch.exp(log_prob))
        loss = -torch.mean(log_l + log_1ml)
        return loss


class ZINB_GNN(nn.Module):
    def __init__(self, gnn_model, zinb_model, n_input, n_hidden, p_dropout):
        super(ZINB_GNN, self).__init__()
        self.gnn_model = gnn_model
        self.zinb_model = zinb_model(n_input, n_hidden, p_dropout)

    def forward(self, x1, x2, x3, bnp1, bnp2, bnp3):
        x = self.gnn_model(x1, x2, x3, bnp1, bnp2, bnp3)
        n, p, pi = self.zinb_model(x)
        return n, p, pi

    def zinb_loss(self, x, n, p, pi):
        return self.zinb_model.zinb_loss(x, n, p, pi)
