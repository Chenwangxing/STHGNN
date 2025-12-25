import torch
import torch.nn as nn
from torch.nn import functional as F
from wtconv import WTConv2d


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module):
    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()
        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.edge_query = nn.Linear(d_model//num_heads, d_model//num_heads)
        self.edge_key = nn.Linear(d_model//num_heads, d_model//num_heads)
        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
    def split_heads(self, x):
        # x [batch_size seq_len d_model]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()
        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]
    def forward(self, x, edge_inital, G):
        # batch_size seq_len 2
        assert len(x.shape) == 3
        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model
        edge_query = self.edge_query(edge_inital)  # batch_size 4 seq_len d_model
        edge_key = self.edge_key(edge_inital)      # batch_size 4 seq_len d_model
        query = self.split_heads(query)  # B num_heads seq_len d_model
        key = self.split_heads(key)  # B num_heads seq_len d_model
        div = torch.sum(G, dim=1)[:, None, :, None]
        Gquery = query + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge_query) / div  # q [batch, num_agent, heads, 64/heads]
        Gkey = key + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge_key) / div
        attention = torch.matmul(Gquery, Gkey.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        attention = self.softmax(attention / self.scaled_factor)
        return attention


class Edge_inital(nn.Module):
    def __init__(self, in_dims=2, d_model=64):
        super(Edge_inital, self).__init__()
        self.x_embedding = nn.Linear(in_dims, d_model//4)
        self.edge_embedding = nn.Linear(d_model//4, d_model//4)
    def forward(self, x, G):
        assert len(x.shape) == 3
        embeddings = self.x_embedding(x)  # batch_size seq_len d_model
        div = torch.sum(G, dim=-1)[:, :, None]
        edge_init = self.edge_embedding(torch.matmul(G, embeddings) / div)  # T N d_model
        edge_init = edge_init.unsqueeze(1).repeat(1, 4, 1, 1)
        return edge_init


class S_Branch(nn.Module):
    def __init__(self):
        super(S_Branch, self).__init__()
        self.tcns = nn.Sequential(nn.Conv2d(8, 8, 1, padding=0),
            nn.PReLU())
        # interaction mask
        self.WTconvolutions = WTConv2d(4, 4, kernel_size=5, wt_levels=3)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        temporal_x = x.permute(1, 0, 2, 3)  # x (num_heads T N N)
        temporal_x = self.tcns(temporal_x) + temporal_x
        x = temporal_x.permute(1, 0, 2, 3)
        threshold = self.activation(self.WTconvolutions(x))
        x_zero = torch.zeros_like(x, device='cuda')
        Sparse_x = torch.where(x > threshold, x, x_zero)
        return Sparse_x


class T_Branch(nn.Module):
    def __init__(self):
        super(T_Branch, self).__init__()
        self.WTconvolutions = WTConv2d(4, 4, kernel_size=5, wt_levels=3)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        threshold = self.activation(self.WTconvolutions(x))
        x_zero = torch.zeros_like(x, device='cuda')
        Sparse_x = torch.where(x > threshold, x, x_zero)
        return Sparse_x


class ST_Branch(nn.Module):
    def __init__(self):
        super(ST_Branch, self).__init__()
        self.WTconvolutions = WTConv2d(4, 4, kernel_size=5, wt_levels=3)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        threshold = self.activation(self.WTconvolutions(x))
        x_zero = torch.zeros_like(x, device='cuda')
        Sparse_x = torch.where(x > threshold, x, x_zero)
        return Sparse_x


class BinaryThresholdFunctionType(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        # 前向传播：应用自适应二值化阈值
        ctx.save_for_backward(input, threshold)
        return (input > 0).float()  # 阈值化操作
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：提供近似梯度
        input, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_threshold = None  # 默认不计算阈值的梯度
        # 对输入张量应用梯度近似
        grad_input[torch.abs(input) > threshold] = 0
        return grad_input, grad_threshold  # 返回梯度


class BinaryThreshold(nn.Module):
    def __init__(self):
        super(BinaryThreshold, self).__init__()
    def forward(self, input, threshold):
        return BinaryThresholdFunctionType.apply(input, threshold)


class STAdaptiveGroupEstimator(nn.Module):
    def __init__(self, in_dims=2):
        super().__init__()
        self.ste = BinaryThreshold()
        self.multi_output = nn.Sequential(nn.Linear(in_dims, 8),
            nn.PReLU(),
            nn.Linear(8, 16))
        self.th = nn.Parameter(torch.Tensor([0.7]))
    def forward(self, node_features):
        # node_features = (T N 2)
        node_features = self.multi_output(node_features)  # node_features (T N 16)
        temp = F.normalize(node_features, p=2, dim=2)  # temp [batch, num_agent, 64]
        corr_mat = torch.matmul(temp, temp.permute(0, 2, 1))  # corr_mat [batch, num_agent, num_agent]
        G = self.ste((corr_mat - self.th.clamp(-0.9999, 0.9999)), self.th.clamp(-0.9999, 0.9999))  # G [batch, num_agent, num_agent]
        return G



class SparseWeightedAdjacency(nn.Module):
    def __init__(self, s_in_dims=2, t_in_dims=3, embedding_dims=64, dropout=0,):
        super(SparseWeightedAdjacency, self).__init__()
        # AdaptiveGroupEstimator
        self.S_Group = STAdaptiveGroupEstimator(in_dims=2)
        self.T_Group = STAdaptiveGroupEstimator(in_dims=3)
        self.ST_Group = STAdaptiveGroupEstimator(in_dims=3)

        # edge_inital
        self.S_edge_inital = Edge_inital(s_in_dims, embedding_dims)
        self.T_edge_inital = Edge_inital(t_in_dims, embedding_dims)
        self.ST_edge_inital = Edge_inital(t_in_dims, embedding_dims)

        # dense interaction
        self.S_group_attention = SelfAttention(s_in_dims, embedding_dims)
        self.T_group_attention = SelfAttention(t_in_dims, embedding_dims)
        self.ST_group_attention = SelfAttention(t_in_dims, embedding_dims)

        self.S_branch = S_Branch()
        self.T_branch = T_Branch()
        self.ST_branch = ST_Branch()

    def forward(self, graph, identity):
        assert len(graph.shape) == 3
        spatial_graph = graph[:, :, 1:]  # (T N 2)
        temporal_graph = graph.permute(1, 0, 2)  # (N T 3)
        st_graph = temporal_graph.contiguous().view(-1, temporal_graph.shape[-1]) # (N*T 3)
        st_graph = st_graph.unsqueeze(0)

        spatial_G = self.S_Group(spatial_graph)  # (T N N)
        temporal_G = self.T_Group(temporal_graph)  # (N T T)
        st_G = self.ST_Group(st_graph)  # (1 N*T N*T)

        S_edge_inital = self.S_edge_inital(spatial_graph, spatial_G)  # (T 4 N 16)
        T_edge_inital = self.T_edge_inital(temporal_graph, temporal_G)  # (N 4 T 16)
        ST_edge_inital = self.ST_edge_inital(st_graph, st_G)  # (1 4 N*T 16)

        group_S_interaction = self.S_group_attention(spatial_graph, S_edge_inital, spatial_G)  # (T num_heads N N)
        group_T_interaction = self.T_group_attention(temporal_graph, T_edge_inital, temporal_G)  # (N num_heads T T)
        group_ST_interaction = self.ST_group_attention(st_graph, ST_edge_inital, st_G)  # (1 num_heads N*T N*T)

        S_interaction = self.S_branch(group_S_interaction)  # (T num_heads N N)
        T_interaction = self.T_branch(group_T_interaction)  # (N num_heads T T)
        ST_interaction = self.ST_branch(group_ST_interaction)  # (1 num_heads N*T N*T)

        S_interaction = S_interaction + identity[0].unsqueeze(1)
        T_interaction = T_interaction + identity[1].unsqueeze(1)
        ST_interaction = ST_interaction + identity[2].unsqueeze(1)

        return (S_interaction, T_interaction, spatial_G, temporal_G, S_edge_inital, T_edge_inital,
                ST_interaction, st_G, ST_edge_inital)


class GraphConvolution(nn.Module):
    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.edge_value = nn.Linear(embedding_dims, in_dims)
        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()
        self.dropout = dropout

    def forward(self, graph, adjacency, G, edge_inital):
        # graph=[T, 1, N, 2](seq_len 1 num_p 2)
        div = torch.sum(G, dim=1)[:, None, :, None]
        edge = self.edge_value(edge_inital)
        value = graph + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge) / div
        gcn_features = self.embedding(torch.matmul(adjacency, value))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)

        return gcn_features  # [batch_size num_heads seq_len hidden_size]


class SparseGraphConvolution(nn.Module):
    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()
        self.dropout = dropout

        self.spatial_gcn = GraphConvolution(in_dims, embedding_dims)
        self.temporal_gcn = GraphConvolution(in_dims, embedding_dims)
        self.spatial_temporal_gcn = GraphConvolution(in_dims, embedding_dims)

        self.weight_attention = nn.Sequential(nn.Linear(embedding_dims * 3, embedding_dims),  # 压缩特征维度
            nn.PReLU(),
            nn.Linear(embedding_dims, 3),  # 输出三个权重
            nn.Softmax(dim=-1))

    def forward(self, graph, S_interaction, T_interaction, spatial_G, temporal_G,
                S_edge_inital, T_edge_inital, ST_interaction, st_G, ST_edge_inital):
        # graph [1 seq_len num_pedestrians  3]
        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # (num_p 1 seq_len 2)

        st_graph = tem_graph.permute(1, 0, 2, 3)
        st_graph = st_graph.contiguous().view(st_graph.shape[0], -1, st_graph.shape[-1])  # (1 N*T 3)
        st_graph = st_graph.unsqueeze(0)

        spatial_features = self.spatial_gcn(spa_graph, S_interaction, spatial_G, S_edge_inital)
        spatial_features = spatial_features.permute(2, 0, 1, 3)
        # spatial_features [N, T, heads, 16]

        temporal_features = self.temporal_gcn(tem_graph, T_interaction, temporal_G, T_edge_inital)
        temporal_features = temporal_features.permute(0, 2, 1, 3)
        # temporal_features [N, T, heads, 16]

        st_features = self.spatial_temporal_gcn(st_graph, ST_interaction, st_G, ST_edge_inital)
        st_features = st_features.reshape([temporal_features.shape[0], temporal_features.shape[1],
                                                 temporal_features.shape[2], temporal_features.shape[3]])

        weights = self.weight_attention(
            torch.cat((spatial_features, temporal_features, st_features), dim=-1))

        weights_features = (weights[..., 0:1]*spatial_features +
                            weights[..., 1:2]*temporal_features +
                            weights[..., 2:3]*st_features)

        fusion_features = spatial_features + temporal_features + st_features

        return weights_features + fusion_features  # [N, T, heads, 16]


class TrajectoryModel(nn.Module):
    def __init__(self,embedding_dims=64, number_gcn_layers=1, dropout=0,obs_len=8, pred_len=12, n_tcn=5, num_heads=4):
        super(TrajectoryModel, self).__init__()

        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout

        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency()

        # graph convolution
        self.stsgcn = SparseGraphConvolution(in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout)

        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(nn.Conv2d(obs_len, pred_len, 3, padding=1),
            nn.PReLU()))
        for j in range(1, self.n_tcn):
            self.tcns.append(nn.Sequential(nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()))

        self.output = nn.Linear(embedding_dims // num_heads, 2)
        self.multi_output = nn.Sequential(nn.Conv2d(num_heads, 16, 1, padding=0),
            nn.PReLU(),
            nn.Conv2d(16, 20, 1, padding=0),)

    def forward(self, graph, identity):
        # graph 1 obs_len N 3   # obs_traj 1 obs_len N 2
        (S_interaction, T_interaction, spatial_G, temporal_G, S_edge_inital, T_edge_inital,
         ST_interaction, st_G, ST_edge_inital) = self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity)

        # gcn_representation = [N, T, heads, 16]
        gcn_representation = self.stsgcn(graph, S_interaction, T_interaction, spatial_G, temporal_G,
                                         S_edge_inital, T_edge_inital, ST_interaction, st_G, ST_edge_inital)

        features = self.tcns[0](gcn_representation)
        for k in range(1, self.n_tcn):
            features = F.dropout(self.tcns[k](features) + features, p=self.dropout)

        prediction = self.output(features)   # prediction=[N, Tpred, nums, 2]
        prediction = self.multi_output(prediction.permute(0, 2, 1, 3))   # prediction=[N, 20, Tpred, 2]

        return prediction.permute(1, 2, 0, 3).contiguous()



