import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k, no_loop=False):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k+int(no_loop), dim=-1)[1]   # (batch_size, num_points, k)
    if no_loop:
        idx = idx[:,:,1:]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda') if x.is_cuda else torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)


    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()  #b*d*n*k
  
    return feature

def attention(x, k=10, idx=None):
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k, no_loop=True)  # (batch_size, num_points, k)
    device = torch.device('cuda') if x.is_cuda else torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    query = x.transpose(2, 1).contiguous()
    key = query.view(batch_size * num_points, -1)[idx, :]
    key = key.view(batch_size, num_points, k, num_dims)  # b*n*k*d
    query = query.view(batch_size, num_points, 1, num_dims)  # b*n*1*d
    value = key - query  # b*n*k*d

    qk = torch.matmul(query, key.permute(0, 1, 3, 2))  # b*n*1*k
    qk = torch.nn.functional.softmax(qk, dim=-1).permute(0, 1, 3, 2)  # b*n*k*1
    feature = qk * value  # b*n*k*d
    feature = torch.sum(feature, dim=-2)  # b*n*d
    res = torch.cat((query.squeeze(), feature), dim=-1)  # b*n*2d
    return  res.permute(0,2,1)  #b*2d*n

class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Attentive_Pooling(nn.Module):
    def __init__(self, args):
        super(Attentive_Pooling, self).__init__()
        self.heads = args.heads
        self.atten_score = nn.Linear(args.emb_dims, args.heads, bias=False)
        # self.atten_bn = nn.LayerNorm(args.emb_dims * args.heads)

    def forward(self,x):
        x = x.permute(0, 2, 1)
        scores = F.softmax(self.atten_score(x), dim=1)  # b*n*h
        atten_out = []  # h*b*d  to save memory
        for i in range(self.heads):
            score = scores[:, :, [i]]  # b*n*1
            atten_out.append(torch.sum(x * score, dim=1))
        atten_out = torch.cat(atten_out, dim=-1)  # list -> tensor  b*(h*d)

        # atten_out = F.leaky_relu(self.atten_bn(atten_out),negative_slope=0.2)

        return  atten_out

class SelfAttention(nn.Module):
    def __init__(self, seq_len, input_dim, emb_dim, value_dim, n_heads, output_dim, attention_mode='scale_dot'):
        super(SelfAttention, self).__init__()
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.query_transforms = nn.ModuleList([nn.Linear(input_dim, emb_dim, bias=False) for _ in range(n_heads)])
        self.key_transforms = nn.ModuleList([nn.Linear(input_dim, emb_dim, bias=False) for _ in range(n_heads)])
        self.value_transforms = nn.ModuleList([nn.Linear(input_dim, value_dim, bias=False) for _ in range(n_heads)])
        self.output_layer = nn.Sequential(nn.Conv1d(value_dim * n_heads, output_dim, kernel_size=1, bias=False),
                                          nn.BatchNorm1d(output_dim),
                                          nn.LeakyReLU(negative_slope=0.2))
        
        if attention_mode == 'dot':
            self.attention_op = lambda q, k: torch.matmul(q, k.permute(0, 1, 3, 2)) # (b,n,k,k)
        elif attention_mode == 'scale_dot':
            self.attention_op = lambda q, k: torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(emb_dim)
        elif attention_mode == 'additive':
            self.additive_layer1 = nn.Linear(emb_dim, value_dim, bias=False)
            self.additive_layer2 = nn.Linear(emb_dim, value_dim, bias=False)
            self.attention_op = lambda q, k: self.additive_layer1(q) + self.additive_layer2(k)

    def get_neigbor_seqence(self, x):
        batch_size = x.size(0)
        num_dims = x.size(1)
        num_points = x.size(2)
        nx = x.view(batch_size, -1, num_points)
        idx = knn(nx, k=self.seq_len, no_loop=True)  # (batch_size, num_points, k)
        device = torch.device('cuda') if x.is_cuda else torch.device('cpu')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        nx = nx.transpose(2, 1).contiguous() # (b, n, d)
        nx = nx.view(batch_size * num_points, -1)[idx, :] 
        nx = nx.view(batch_size, num_points, self.seq_len, num_dims).contiguous() # (b,n,k,d)
        return nx

    def forward(self, x):
        neighbors = self.get_neigbor_seqence(x)

        xx = x.transpose(2, 1).contiguous()  # (b, d, n) -> (b, n, d)
        q = xx.unsqueeze(2) # (b, n, d) -> (b, n, 1, d)
        k = neighbors
        v = neighbors - xx.unsqueeze(2)
        heads_out = []
        for i in range(self.n_heads):
            query =  self.query_transforms[i](q)
            key = self.key_transforms[i](k)
            value = self.value_transforms[i](v)

            scores = F.softmax(self.attention_op(query, key), dim=-1)
            out = torch.mean(torch.matmul(scores, value), dim=2)
            heads_out.append(out)
        
        mix = torch.cat(heads_out, dim=-1).permute(0, 2, 1).contiguous() # (b, n, e) -> (b, e, n)
        result = self.output_layer(mix)  # (b, e, n) -> (b, d, n)
        return torch.cat([result, x], dim=1)


class Mymodel(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Mymodel, self).__init__()
        self.args = args
        self.k = args.k
        self.att_pooling = Attentive_Pooling(args)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv1d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.atten_out_linear = nn.Linear(args.emb_dims*args.heads,args.emb_dims*2)
        self.linear1 = nn.Linear(args.emb_dims*2,512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.self_att1 = SelfAttention(args.k, 3, 3, 3, 1, 3)
        self.self_att2 = SelfAttention(args.k, 64, 64, 64, 1, 64)
        self.self_att3 = SelfAttention(args.k, 64, 64, 64, 1, 64)
        self.self_att4 = SelfAttention(args.k, 128, 128, 128, 1, 128)

    def forward(self, x):
        batch_size = x.size(0)
        # x = attention(x, k=self.k)
        x = self.self_att1(x)
        x1 = self.conv1(x)

        # x = attention(x1, k=self.k)
        x = self.self_att2(x1)
        x2 = self.conv2(x)

        x = self.self_att3(x2)
        x3 = self.conv3(x)

        x = self.self_att4(x3)
        x4 = self.conv4(x)  #b * d * n

        x = torch.cat((x1, x2, x3, x4), dim=1) #b*512*n
        x = self.conv5(x)

        if self.args.pooling == 'max':
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            pooling_out = torch.cat((x1, x2), 1)
        else:
            atten_out = self.att_pooling(x)
            pooling_out = self.atten_out_linear(atten_out)

        x = F.leaky_relu(self.bn6(self.linear1(pooling_out)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class DGANetSpectral(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGANetSpectral, self).__init__()
        self.args = args
        self.k = args.k
        self.att_pooling = Attentive_Pooling(args)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv1d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims,512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.self_att1 = SelfAttention(args.k, 3, 3, 3, 1, 3)
        self.self_att2 = SelfAttention(args.k, 64, 64, 64, 1, 64)
        self.self_att3 = SelfAttention(args.k, 64, 64, 64, 1, 64)
        self.self_att4 = SelfAttention(args.k, 128, 128, 128, 1, 128)

        device = torch.device('cuda') if args.cuda else torch.device('cpu')
        self.k_cluster = args.k_cluster
        self.temperature = args.temperature
        # init_clustering = torch.normal(0, 1/ np.sqrt(args.num_points), (args.num_points, args.k_cluster), requires_grad=True)
        # self.clustering = nn.Parameter(init_clustering)
        self.gen_clustering = nn.Conv1d(args.emb_dims, self.k_cluster, kernel_size=1)
        self.identity_mat = torch.eye(self.k_cluster, device=device) / np.sqrt(self.k_cluster)

    def spectral_pooling(self, x):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        adj = torch.exp((-xx - inner - xx.transpose(2, 1)) / 10)
        deg_diag = torch.sum(adj, dim=-1)
        deg_mat = torch.diag_embed((deg_diag + 1e-10) ** (-0.5))
        laplacian = torch.matmul(torch.matmul(deg_mat, adj), deg_mat)

        clustering = F.softmax(self.gen_clustering(x) / self.temperature, dim=1)
        clustering = clustering.transpose(2,1).contiguous() # b, n, k

        trace = lambda tensor: torch.diagonal(tensor, dim1=1, dim2=2).sum(axis=-1)
        cut_cost = trace(torch.matmul(torch.matmul(clustering.transpose(2,1), laplacian), clustering))
        max_cut_cost = trace(torch.matmul(torch.matmul(clustering.transpose(2,1), torch.diag_embed(deg_diag)), clustering))
        cut_loss = -torch.mean(cut_cost / max_cut_cost)

        gram = torch.matmul(clustering.transpose(2, 1), clustering) # (b, k, k)
        gram = gram / (torch.linalg.norm(gram, dim=(1,2), ord='fro') + 1e-10).reshape(-1, 1, 1)
        orthog_diff = gram - self.identity_mat
        orthog_loss = torch.mean(torch.linalg.norm(orthog_diff, dim=(1,2), ord='fro'))

        out = torch.matmul(x, clustering)
        out = torch.max(out, axis=-1)[0]

        return out, cut_loss + orthog_loss

    def forward(self, x):
        batch_size = x.size(0)
        # x = attention(x, k=self.k)
        x = self.self_att1(x)
        x1 = self.conv1(x)

        # x = attention(x1, k=self.k)
        x = self.self_att2(x1)
        x2 = self.conv2(x)

        x = self.self_att3(x2)
        x3 = self.conv3(x)

        x = self.self_att4(x3)
        x4 = self.conv4(x)  #b * d * n

        x = torch.cat((x1, x2, x3, x4), dim=1) #b*512*n
        x = self.conv5(x)

        out, clustering_loss = self.spectral_pooling(x)

        x = F.leaky_relu(self.bn6(self.linear1(out)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x, clustering_loss

