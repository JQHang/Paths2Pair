import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 残差网络，用于转换特征，以及输出最终预测结果
class DNN(nn.Module):
    def __init__(self, input_size, output_size, dropout = 0.5):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(output_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class Res_DNN_layer(nn.Module):
    def __init__(self, hidden_size, dropout, num_DNN):
        super().__init__()
        self.multi_DNN = nn.ModuleList([DNN(hidden_size, hidden_size, dropout) for _ in range(num_DNN)])
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        
        hidden_states_shortcut = hidden_states
        for i,layer_module in enumerate(self.multi_DNN):
            hidden_states = layer_module(hidden_states)
        hidden_states = hidden_states_shortcut + hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states

class Res_DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_Res_DNN, num_DNN):
        super().__init__()
        # 先将数据降维
        self.prepare = nn.Linear(input_size, hidden_size) 
        
        # 再导入两轮3层Res_DNN
        self.multi_Res_DNN = nn.ModuleList([Res_DNN_layer(hidden_size, dropout, num_DNN) for _ in range(num_Res_DNN)])
        
        # 输出层，简单的一个线性层，从hidden_size映射到num_labels
        self.classifier = nn.Linear(hidden_size, output_size) 
        
    def forward(self, input_ids):
        hidden_states = self.prepare(input_ids)
        
        for i,layer_module in enumerate(self.multi_Res_DNN):
            hidden_states = layer_module(hidden_states)
        
        hidden_states = self.classifier(hidden_states)
    
        return hidden_states

# 添加position embedding(先加入pos embedding，再标准化，再dropout)
class PositionEmbeddings(nn.Module):
    def __init__(self, nfeat, seq_length, dropout):
        super().__init__()
        
        self.seq_length = seq_length
        
        self.position_embeddings = nn.Embedding(seq_length, nfeat)
        
        self.LayerNorm = nn.LayerNorm(nfeat)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, features_embeddings):
        
        position_embeddings = self.position_embeddings.weight.unsqueeze(1).expand(self.seq_length, features_embeddings.size(1), 
                                                          features_embeddings.size(2))
        
        embeddings = features_embeddings + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        
        embeddings = self.dropout(embeddings)
        
        return embeddings

class TransformerLayer(nn.Module):
    def __init__(self, nfeat, nhead, nhid, nout, nlayers, seq_length, dropout=0.5):
        super().__init__()
        
        self.feature_Linear = nn.Linear(nfeat, nhid)
        
        # self.get_embed = PositionEmbeddings(nfeat = nhid, seq_length = seq_length, dropout = dropout)
        
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.decoder = nn.Linear(nhid*seq_length, nout)
            
    def forward(self, h):
        h = self.feature_Linear(h)
        
        # h = self.get_embed(h)
        
        h = self.transformer_encoder(h)

        batch_num = h.size(1)
        h = self.decoder(h.permute(1, 0, 2).reshape(batch_num, -1))
    
        return h
    
class SeHGNN(nn.Module):
    def __init__(self, Node_Type_to_Feature_len_dict, All_Meta_Path_Name_list, Meta_Path_Column_Type_dict, node_feature_hid_len, 
             metapath_level_nhid, metapath_level_nhead, metapath_level_nlayers, num_Res_DNN, each_Res_DNN_num, dropout=0.5):
        super().__init__()
        
        # 随机dropout函数，防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 对原始特征进行映射
        self.node_feature_transform_dict = {}
        for tmp_node_type in Node_Type_to_Feature_len_dict:
#             tmp = nn.Linear(Node_Type_to_Feature_len_dict[tmp_node_type], node_feature_hid_len)
            tmp = Res_DNN(Node_Type_to_Feature_len_dict[tmp_node_type], node_feature_hid_len, node_feature_hid_len, dropout, 
                      num_Res_DNN, each_Res_DNN_num)
            
            self.node_feature_transform_dict[tmp_node_type] = tmp
            self.add_module('Node_feature_transform_{}'.format(tmp_node_type), tmp)
        
        self.Meta_Path_Column_Type_dict = Meta_Path_Column_Type_dict

        # 用于预测的transformer
#         tmp_meta_path_node_count = 0
#         for tmp_meta_path_name in Meta_Path_Column_Type_dict:
#             tmp_meta_path_node_count = tmp_meta_path_node_count + len(Meta_Path_Column_Type_dict[tmp_meta_path_name].keys())
#         self.transformer = TransformerLayer(node_feature_hid_len, metapath_level_nhead, metapath_level_nhid, metapath_level_nhid,
#                                 metapath_level_nlayers, tmp_meta_path_node_count + 1, dropout)
        
        # 用于预测的transformer
        self.transformer = TransformerLayer(node_feature_hid_len, metapath_level_nhead, metapath_level_nhid, metapath_level_nhid,
                                metapath_level_nlayers, len(All_Meta_Path_Name_list), dropout)
        
        # 最后的输出函数
        self.output_dense = Res_DNN(metapath_level_nhid, metapath_level_nhid, 1, dropout, num_Res_DNN, each_Res_DNN_num)
#         self.output_dense = nn.Linear(semantic_level_nhid, 1)
        
        self.activation = nn.Sigmoid()
        
    def forward(self, input_feature_dict):
        # 获取各关系对应的特征
        metapath_h_feature_list = []
    
        # 再对元路径特征进行转换
        for tmp_meta_path_name in input_feature_dict['Feature_Dict']:
#             print(tmp_meta_path_name)
            
#             for tmp_index in input_feature_dict['Meta_Path_Feature'][tmp_meta_path_name]:
#                 tmp_node_type = self.Meta_Path_Column_Type_dict[tmp_meta_path_name][tmp_index]

#                 tmp_transferred_h = self.node_feature_transform_dict[tmp_node_type](input_feature_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_index])
            
#                 metapath_h_feature_list.append(tmp_transferred_h)

            tmp_node_type = self.Meta_Path_Column_Type_dict[tmp_meta_path_name]

            tmp_transferred_h = self.node_feature_transform_dict[tmp_node_type](input_feature_dict['Feature_Dict'][tmp_meta_path_name])

            metapath_h_feature_list.append(tmp_transferred_h)

        ###################################################################################################################
        # 合并各元路径的结果
        tmp_metapath_h_feature_stack = torch.stack(metapath_h_feature_list, 0)
        tmp_metapath_h_feature_stack = self.dropout(tmp_metapath_h_feature_stack)
            
        # 通过semantic_level transformer
        tmp_metapath_h_feature_stack = self.transformer(tmp_metapath_h_feature_stack)
            
        # 输出最终结果
        h_output = self.output_dense(tmp_metapath_h_feature_stack)
        h_output = h_output.squeeze()
        h_output = self.activation(h_output)
        
        return h_output