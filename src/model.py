import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import GCN
import math
from utils import to_3d, to_4d


class Redumixdti(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.drug_extractor = MoleculeGCN(configs)
        self.prot_extractor = ProteinCNN(configs)
        self.ema_d = EMARU(configs.Drug.Max_Nodes, configs.MBCA.Hidden_Size, gate_threshold=0.3)
        self.ema_p = EMARU(configs.Protein.After_CNN_Length, configs.MBCA.Hidden_Size, gate_threshold=0.3)
        self.fusion = MBCA(configs)
        self.mlp_classifier = DropoutMLP(configs)

    def forward(self, d_graph, p_feat, mode='train'):
        v_d = self.drug_extractor(d_graph)
        v_p = self.prot_extractor(p_feat)
        v_d = v_d + to_3d(self.ema_d(to_4d(v_d, v_d.size(1), 1)))
        v_p = v_p + to_3d(self.ema_p(to_4d(v_p, v_p.size(1), 1)))
        f, attn = self.fusion(v_d, v_p)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, attn, score


class MoleculeGCN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.in_feat = configs.Drug.Node_In_Feat
        self.dim_embedding = configs.Drug.Node_In_Embedding
        self.hidden_feats = configs.Drug.Hidden_Layers
        self.padding = configs.Drug.Padding
        self.activation = configs.Drug.GCN_Activation

        self.init_linear = nn.Linear(self.in_feat, self.dim_embedding, bias=False)
        if self.padding:
            with torch.no_grad():
                self.init_linear.weight[-1].fill_(0)
        self.gcn = GCN(in_feats=self.dim_embedding, hidden_feats=self.hidden_feats, activation=self.activation)
        self.output_feats = self.hidden_feats[-1]

    def forward(self, batch_d_graph):
        node_feats = batch_d_graph.ndata['h']
        node_feats = self.init_linear(node_feats)
        node_feats = self.gcn(batch_d_graph, node_feats)
        batch_size = batch_d_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinCNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.embedding_dim = configs.Protein.Embedding_Dim
        self.num_filters = configs.Protein.Num_Filters
        self.kernel_size = configs.Protein.Kernel_Size
        self.padding = configs.Protein.Padding

        if self.padding:
            self.embedding = nn.Embedding(26, self.embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, self.embedding_dim)
        in_out_ch = [self.embedding_dim] + self.num_filters
        kernels = self.kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_out_ch[0], out_channels=in_out_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_out_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_out_ch[1], out_channels=in_out_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_out_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_out_ch[2], out_channels=in_out_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_out_ch[3])

    def forward(self, p_feat):
        p_feat = self.embedding(p_feat.long())
        p_feat = p_feat.transpose(2, 1)
        p_feat = F.relu(self.bn1(self.conv1(p_feat)))
        p_feat = F.relu(self.bn2(self.conv2(p_feat)))
        p_feat = F.relu(self.bn3(self.conv3(p_feat)))
        p_feat = p_feat.transpose(2, 1)
        return p_feat


def reconstruct(x_1, x_2):
    x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
    x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
    return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class EMARU(nn.Module):
    def __init__(self,
                 avg_len,
                 channels,
                 factor=8,
                 gate_threshold: float = 0.5,
                 ):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AvgPool2d((avg_len, 1))
        self.pool_h = nn.AvgPool2d((1, 1))
        self.pool_w = nn.AvgPool2d((avg_len, 1))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1)

        self.gate_threshold = gate_threshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        reweights = group_x * self.sigomid(weights)
        info_mask = reweights >= self.gate_threshold
        noninfo_mask = reweights < self.gate_threshold
        x_1 = info_mask * group_x
        x_2 = noninfo_mask * group_x
        x = reconstruct(x_1, x_2).reshape(b, c, h, w)
        return x


class MBCA(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.positional_drug = PositionalEncoding(configs.MBCA.Hidden_Size, max_len=configs.Drug.Max_Nodes)
        self.positional_prot = PositionalEncoding(configs.MBCA.Hidden_Size, max_len=configs.Protein.After_CNN_Length)
        self.attn_map = AttenMapNHeads(configs)

        self.attention_fc_dp = nn.Linear(configs.MBCA.Num_Heads, configs.MBCA.Hidden_Size)
        self.attention_fc_pd = nn.Linear(configs.MBCA.Num_Heads, configs.MBCA.Hidden_Size)

    def forward(self, drug, protein):
        drug = self.positional_drug(drug)
        protein = self.positional_prot(protein)

        attn_map = self.attn_map(drug, protein)
        att_dp = F.softmax(attn_map, dim=-1)
        att_pd = F.softmax(attn_map, dim=-2)
        attn_matrix = 0.5 * att_dp + 0.5 * att_pd

        drug_attn = self.attention_fc_dp(torch.mean(attn_matrix, -1).transpose(-1, -2))
        protein_attn = self.attention_fc_pd(torch.mean(attn_matrix, -2).transpose(-1, -2))

        drug_attn = F.sigmoid(drug_attn)
        protein_attn = F.sigmoid(protein_attn)

        drug = drug + drug * drug_attn
        protein = protein + protein * protein_attn

        drug, _ = torch.max(drug, 1)
        protein, _ = torch.max(protein, 1)

        pair = torch.cat([drug, protein], dim=1)
        return pair, (drug_attn, protein_attn)


class DropoutMLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(configs.MLP.In_Dim * 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, configs.MLP.Binary)

    def forward(self, pair):
        pair = self.dropout1(pair)
        fully1 = F.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = F.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = F.leaky_relu(self.fc3(fully2))
        pred = self.out(fully3)
        return pred


class AttenMapNHeads(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.hid_dim = configs.MBCA.Hidden_Size
        self.n_heads = configs.MBCA.Num_Heads

        assert self.hid_dim % self.n_heads == 0

        self.f_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.f_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.d_k = self.hid_dim // self.n_heads

    def forward(self, d, p):
        batch_size = d.shape[0]

        Q = self.f_q(d)
        K = self.f_k(p)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        return attn_weights


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1), :]
