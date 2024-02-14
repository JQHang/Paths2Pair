import numpy as np
import pandas as pd
import os

from kg_lib.utils import mkdir

"""
作用：
    获得能符合subgraph类型模型要求的pandas数据（包含标签列 + 元路径 + 各节点对应的特征），主要是得重新为各节点生成序号，并按新的序号修正关系表以及特征表。

输入：
    sample_source_str：目标节点来源的字符串描述（为了定位元路径、节点、特征的存储位置）
    time_range_str：目标节点对应的时间区间或点的字符串描述（也是为了定位元路径、节点、特征的存储位置）
    aim_node_type：目标点类型
    subgraph_hop_num:要处理几跳的subgraph
    Relation_drop_list:不需要的关系名称列表，会自动跳过对应关系
    Feature_Type_list:使用的特征类型（可以使用Raw、Norm、Std）
    
返回值：
    输出结果直接存储在对应文件夹中
"""
def get_subgraph_required_pandas_data(sample_source_str, time_range_str, aim_node_type, subgraph_hop_num, regenerate = False,
                          Relation_drop_list = [], Feature_Type_list = ['Norm']):
    Processed_Subgraph_Data_dict = {}
    Processed_Subgraph_Data_dict['Feature'] = {}
    Processed_Subgraph_Data_dict['Adj'] = {}
    
    tmp_all_output_data_base_dir = '../Data/'
    tmp_all_output_data_time_range_dir = ('../Data/' + sample_source_str + '/' +  time_range_str + '/')
    
    print('预处理' + tmp_all_output_data_time_range_dir + '文件夹下的相关文件')

    # 设置目标关系表存储文件夹
    tmp_all_output_data_time_range_subgraph_dir = tmp_all_output_data_time_range_dir + 'Subgraph/Edges/'
    
    # 设置节点存储文件夹
    tmp_all_output_data_time_range_node_dir = tmp_all_output_data_time_range_dir + 'Node/'
    
    # 设置节点特征存储文件夹
    tmp_all_output_data_time_range_feature_dir = tmp_all_output_data_time_range_dir + 'Feature/'
    
    # 读取标签表
    label_pd = pd.read_pickle(tmp_all_output_data_time_range_dir + 'Target_Node.pkl')
    
    # 获取目标列列名
    aim_node_column_name = aim_node_type + '_UID'

    # 只保留其中非标签的列，作为目标节点列表
    if 'Label' in label_pd.columns:
        aim_UID_pd = label_pd[[aim_node_column_name, 'Label']].copy()
    else:
        aim_UID_pd = label_pd[[aim_node_column_name]].copy()
    
    #######################################################################################################################
    # 设置subgraph_GNN所需的数据的存储位置
    tmp_all_output_data_time_range_subgraph_GNN_dir = tmp_all_output_data_time_range_dir + 'Subgraph_GNN/'
    mkdir(tmp_all_output_data_time_range_subgraph_GNN_dir)
    
    # 建立subgraph所需的特征的存储文件夹
    tmp_all_output_data_time_range_subgraph_GNN_Feature_dir = tmp_all_output_data_time_range_subgraph_GNN_dir + 'Feature/'
    mkdir(tmp_all_output_data_time_range_subgraph_GNN_Feature_dir)
    
    # 建立subgraph所需的邻接表的存储文件夹
    tmp_all_output_data_time_range_subgraph_GNN_Adj_dir = tmp_all_output_data_time_range_subgraph_GNN_dir + 'Adj/'
    mkdir(tmp_all_output_data_time_range_subgraph_GNN_Adj_dir)
    
    #######################################################################################################################
    # 获取节点表包含的全部文件名，看看有多少类型的节点
    tmp_node_table_file_list = os.listdir(tmp_all_output_data_time_range_node_dir)
    tmp_node_type_list = [x.split('.pkl')[0] for x in tmp_node_table_file_list if '.pkl' in x]
    print('涉及到的全部节点类型:', tmp_node_type_list)
    
    # 依次读取各节点的特征表，将各节点在节点表中的index作为新序号
    tmp_node_UID_to_index_dict = {}
    for tmp_node_type in tmp_node_type_list:
        print('处理节点:', tmp_node_type)
        
        # 查看是否已处理过，若已处理过，则跳过处理，直接读取信息
        if not regenerate and os.path.exists(tmp_all_output_data_time_range_subgraph_GNN_Feature_dir + tmp_node_type + '.pkl'):
            print(tmp_node_type + '节点特征已处理过，直接读取')
            tmp_node_all_feature_pd = pd.read_pickle(tmp_all_output_data_time_range_subgraph_GNN_Feature_dir + tmp_node_type + '.pkl')
            
            # 只保留特征，不要UID
            Processed_Subgraph_Data_dict['Feature'][tmp_node_type]= tmp_node_all_feature_pd.drop(columns=[tmp_node_type + '_UID']).values
            
            # 获取UID到index的对应关系，作为各节点的数值序号
            tmp_UID_to_Index_pd = tmp_node_all_feature_pd[[tmp_node_type + '_UID']].reset_index()
            tmp_node_UID_to_index_dict[tmp_node_type] = tmp_UID_to_Index_pd
            
            continue
        
        # 读取涉及的各类型节点表，并合并
        tmp_all_feature_pd_list = []
        for tmp_feature_type in Feature_Type_list:
            # 获取特征文件位置(不会处理过去月份的特征，故只用0号文件夹即可)
            tmp_node_feature_file = tmp_all_output_data_time_range_feature_dir + '0/' + tmp_node_type + '_' + tmp_feature_type + '.pkl'

            # 读取特征文件
            tmp_node_feature_pd = pd.read_pickle(tmp_node_feature_file)

            # 修正列名(除了目标列，都加上后缀)
            tmp_node_feature_pd.columns = tmp_node_feature_pd.columns.map(lambda x: x + '_' + tmp_feature_type if x != (tmp_node_type + '_UID') else x)

            # 如果不是第一个特征表，就删去UID列
            if len(tmp_all_feature_pd_list) != 0:
                tmp_node_feature_pd = tmp_node_feature_pd.drop(columns=[tmp_node_type + '_UID'])
            
            tmp_all_feature_pd_list.append(tmp_node_feature_pd)
            
        # 合并各个特征表(因为不同类型的特征表，各行对应的UID一定是一致的)
        tmp_node_all_feature_pd = pd.concat(tmp_all_feature_pd_list, axis = 1)
        
        # 去重(理论上用不到，特征表阶段应该已经去过重了，但以防万一)
        tmp_node_all_feature_pd = tmp_node_all_feature_pd.drop_duplicates([tmp_node_type + '_UID'], ignore_index = True)
        
        # 保存节点特征到返回信息中
        Processed_Subgraph_Data_dict['Feature'][tmp_node_type] = tmp_node_all_feature_pd.drop(columns=[tmp_node_type + '_UID']).values
        
        # 获取UID到index的对应关系，作为各节点的数值序号
        tmp_UID_to_Index_pd = tmp_node_all_feature_pd[[tmp_node_type + '_UID']].reset_index()
        tmp_node_UID_to_index_dict[tmp_node_type] = tmp_UID_to_Index_pd
        
        # 存储拼接后的特征
        tmp_node_all_feature_pd.to_pickle(tmp_all_output_data_time_range_subgraph_GNN_Feature_dir + tmp_node_type + '.pkl')
    
    #######################################################################################################################
    # 查看是否已处理过标签表，若已处理过，则跳过处理，直接读取信息
    if not regenerate and os.path.exists(tmp_all_output_data_time_range_subgraph_GNN_Feature_dir + 'Target_Node_with_Index.pkl'):
        aim_UID_with_new_index_pd = pd.read_pickle(tmp_all_output_data_time_range_subgraph_GNN_Feature_dir + 'Target_Node_with_Index.pkl')
        
        Processed_Subgraph_Data_dict['Target_Node_Index'] = aim_UID_with_new_index_pd['index'].values
        
        if 'Label' in aim_UID_with_new_index_pd.columns:
            Processed_Subgraph_Data_dict['Target_Node_Label'] = aim_UID_with_new_index_pd['Label'].values
    else:
        # 按新序号重新排列标签表中的标签序号
        aim_UID_with_new_index_pd = aim_UID_pd.merge(tmp_node_UID_to_index_dict[aim_node_type], on = aim_node_type + '_UID', 
                                      how = 'left').reset_index(drop = True)

        # 存储有标签的index号，以及对应的顺序
        aim_UID_with_new_index_pd.to_pickle(tmp_all_output_data_time_range_subgraph_GNN_Feature_dir + 'Target_Node_with_Index.pkl')

        # 保存标签信息到返回信息中
        Processed_Subgraph_Data_dict['Target_Node_Index'] = aim_UID_with_new_index_pd['index'].values
        
        # 确认全部index都不为nan
        if np.sum(np.isnan(Processed_Subgraph_Data_dict['Target_Node_Index'])) != 0:
            print('ERROR: Label_Index中nan值数为:', np.sum(np.isnan(Processed_Subgraph_Data_dict['Target_Node_Index'])))
        
        if 'Label' in aim_UID_with_new_index_pd.columns:
            Processed_Subgraph_Data_dict['Target_Node_Label'] = aim_UID_with_new_index_pd['Label'].values
            
    #######################################################################################################################
    # 依次处理各跳的关系
    for tmp_hop in range(subgraph_hop_num):
        print('开始处理第', tmp_hop, '跳的数据')
        
        Processed_Subgraph_Data_dict['Adj'][tmp_hop] = {}
        
        # 设置各跳的结果输出文件夹
        tmp_subgraph_GNN_Adj_current_hop_processed_dir = tmp_all_output_data_time_range_subgraph_GNN_Adj_dir + str(tmp_hop) + '/'
        mkdir(tmp_subgraph_GNN_Adj_current_hop_processed_dir)
        
        # 获取该跳中包含的关系名
        tmp_subgraph_edges_current_hop_output_dir = tmp_all_output_data_time_range_subgraph_dir + str(tmp_hop) + '/'
        tmp_relation_file_list = os.listdir(tmp_subgraph_edges_current_hop_output_dir)
        tmp_relation_file_list = [tmp_file_name for tmp_file_name in tmp_relation_file_list if '.pkl' in tmp_file_name]
        
        # 依次读取各关系对应的数据
        for tmp_relation_file in tmp_relation_file_list:
            tmp_relation_name = tmp_relation_file.split('_Aim_')[0]
            tmp_aim_head_or_tail = tmp_relation_file.split('.pkl')[0].split('_Aim_')[1]
            tmp_relation_name_with_aim = tmp_relation_file.split('.pkl')[0]

#             print(tmp_relation_name_with_aim)
#             print(Relation_drop_list)
#             print(tmp_relation_name_with_aim in Relation_drop_list)
            
            # 跳过不需要的元路径
            if tmp_relation_name_with_aim in Relation_drop_list:
                print('跳过关系:', tmp_relation_name_with_aim)
                continue
        
            # 读取元路径对应的节点类型
            tmp_column_to_node_class_dict = np.load(tmp_subgraph_edges_current_hop_output_dir + tmp_relation_name
                                       + '-column_to_node_class.npy', allow_pickle= True).item()
            
            if tmp_aim_head_or_tail == 'Head':
                head_column = 'Head'
                tail_column = 'Tail'
            else:
                head_column = 'Tail'
                tail_column = 'Head'
            
            # 获取首尾两列对应的节点类型
            head_column_node_type = tmp_column_to_node_class_dict[head_column]
            tail_column_node_type = tmp_column_to_node_class_dict[tail_column]

            Processed_Subgraph_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim] = {}
            Processed_Subgraph_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['head_type'] = head_column_node_type
            Processed_Subgraph_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['tail_type'] = tail_column_node_type

            # 查看是否已处理过对应邻接表，若已处理过，则跳过处理，直接读取信息
            if not regenerate and os.path.exists(tmp_subgraph_GNN_Adj_current_hop_processed_dir + tmp_relation_name_with_aim + '.pkl'):
                print(tmp_relation_name_with_aim + '对应邻接表已处理过，直接读取')
                tmp_edge_result_pandas = pd.read_pickle(tmp_subgraph_GNN_Adj_current_hop_processed_dir + tmp_relation_name_with_aim + '.pkl')
                Processed_Subgraph_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Adj']= tmp_edge_result_pandas.values.T
                continue

            # 读取元路径信息
            tmp_edge_result_pandas = pd.read_pickle(tmp_subgraph_edges_current_hop_output_dir + tmp_relation_file)
            print('读取关系:', tmp_relation_name_with_aim)

            # 只保留首尾两列
            tmp_edge_result_pandas = tmp_edge_result_pandas[[head_column, tail_column]]

            # 获取首列对应的index
            tmp_edge_result_pandas = tmp_edge_result_pandas.merge(tmp_node_UID_to_index_dict[head_column_node_type], left_on = head_column,
                                                       right_on = head_column_node_type + '_UID', how = 'left')

            tmp_edge_result_pandas = tmp_edge_result_pandas.drop(columns=[head_column_node_type + '_UID'])

            tmp_edge_result_pandas = tmp_edge_result_pandas.rename(columns={'index' : 'head'})

            # 确认首列中没有nan值
            if np.sum(np.isnan(tmp_edge_result_pandas['head'].values)) != 0:
                print('ERROR: 该关系首列中nan值数为:', np.sum(np.isnan(tmp_edge_result_pandas['head'].values)))

            # 获取尾列对应的index
            tmp_edge_result_pandas = tmp_edge_result_pandas.merge(tmp_node_UID_to_index_dict[tail_column_node_type], left_on = tail_column,
                                                right_on = tail_column_node_type + '_UID', how = 'left')

            tmp_edge_result_pandas = tmp_edge_result_pandas.drop(columns=[tail_column_node_type + '_UID'])

            tmp_edge_result_pandas = tmp_edge_result_pandas.rename(columns={'index' : 'tail'})

            # 确认尾列中没有nan值
            if np.sum(np.isnan(tmp_edge_result_pandas['tail'].values)) != 0:
                print('ERROR: 该关系尾列中nan值数为:', np.sum(np.isnan(tmp_edge_result_pandas['tail'].values)))

            # 只保留首尾两列对应的index
            tmp_edge_result_pandas = tmp_edge_result_pandas[['head', 'tail']]

            # 保存邻接表信息到返回信息中
            Processed_Subgraph_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Adj'] = tmp_edge_result_pandas.values.T

            # 存储结果
            tmp_edge_result_pandas.to_pickle(tmp_subgraph_GNN_Adj_current_hop_processed_dir + tmp_relation_name_with_aim + '.pkl')

    return Processed_Subgraph_Data_dict