import numpy as np
import pandas as pd
import os
import torch

from kg_lib.utils import mkdir

"""
作用：
    获得能符合MAGNN模型要求的pandas数据（包含标签列 + 元路径 + 各节点对应的特征），主要是得重新为各节点生成序号，并按新的序号修正关系表以及特征表。

输入：
    sample_source_str：目标节点来源的字符串描述（为了定位元路径、节点、特征的存储位置）
    time_range_str：目标节点对应的时间区间或点的字符串描述（也是为了定位元路径、节点、特征的存储位置）
    Meta_path_drop_list:不需要的元路径名称列表，会自动删去对应元路径
    Feature_Type_list:使用的特征类型（可以使用Raw、Norm、Std）
    
返回值：
    输出结果直接存储在对应文件夹中，同时返回用字典存储的标签+关系+特征
"""
def Get_MAGNN_Required_Data_and_Store(sample_source_str, time_range_str, aim_node_type, Meta_path_drop_list = [], Feature_Type_list = ['Norm']):
    Result_Data_dict = {}
    Result_Data_dict['Feature'] = {}
    Result_Data_dict['Adj'] = {}
    Result_Data_dict['Adj_Node_Type'] = {}
    
    tmp_all_output_data_base_dir = '../../Data/'
    tmp_all_output_data_time_range_dir = ('../../Data/' + sample_source_str + '/' +  time_range_str + '/')
    
    print('预处理' + tmp_all_output_data_time_range_dir + '文件夹下的相关文件')
    
    # 读取标签表
    label_pd = pd.read_pickle(tmp_all_output_data_time_range_dir + 'Target_Node.pkl')

    # 获取目标列列名
    aim_node_column_name = aim_node_type + '_UID'
    
    if aim_node_column_name not in label_pd.columns:
        print('Error:目标节点表中没有目标节点对应列名' + aim_node_column_name)
        return
    
    # 只保留其中非标签的列，作为目标节点列表(如果有label则一并保存)
    if 'Label' in label_pd.columns:
        aim_UID_pd = label_pd[[aim_node_column_name, 'Label']].copy()
    else:
        aim_UID_pd = label_pd[[aim_node_column_name]].copy()
        
    # 设置目标元路径存储文件夹
    tmp_all_output_data_time_range_meta_path_dir = tmp_all_output_data_time_range_dir + 'Meta_Path/'
    
    # 设置节点特征存储文件夹
    tmp_all_output_data_time_range_feature_dir = tmp_all_output_data_time_range_dir + 'Feature/0/'
    
    #######################################################################################################################
    # 设置MAGNN所需的数据的存储位置
    tmp_all_output_data_time_range_Model_dir = tmp_all_output_data_time_range_dir + 'MAGNN/'
    mkdir(tmp_all_output_data_time_range_Model_dir)
    
    # 建立MAGNN所需的特征的存储文件夹
    tmp_all_output_data_time_range_Model_Feature_dir = tmp_all_output_data_time_range_Model_dir + 'Feature/'
    mkdir(tmp_all_output_data_time_range_Model_Feature_dir)
    
    # 建立MAGNN所需的邻接表的存储文件夹
    tmp_all_output_data_time_range_Model_Adj_dir = tmp_all_output_data_time_range_Model_dir + 'Adj/'
    mkdir(tmp_all_output_data_time_range_Model_Adj_dir)
    
    #######################################################################################################################
    # 获取节点表包含的全部文件名，看看有多少类型的节点
    tmp_node_feature_table_file_list = os.listdir(tmp_all_output_data_time_range_feature_dir)
    tmp_node_type_list = [x.split('_Raw.pkl')[0] for x in tmp_node_feature_table_file_list if '_Raw.pkl' in x]
    print('涉及到的全部节点类型:', tmp_node_type_list)
    
    # 依次读取各节点的特征表，将各节点在特征表中的index作为新序号
    tmp_node_UID_to_index_dict = {}
    for tmp_node_type in tmp_node_type_list:
        print('处理节点:', tmp_node_type)
        
        # 查看是否已处理过，若已处理过，则跳过处理，直接读取信息
        if os.path.exists(tmp_all_output_data_time_range_Model_Feature_dir + tmp_node_type + '_Feature.pkl'):
            print(tmp_node_type + '节点特征已处理过，直接读取')
            tmp_node_all_feature_pd = pd.read_pickle(tmp_all_output_data_time_range_Model_Feature_dir + tmp_node_type + '_Feature.pkl')
            
            Result_Data_dict['Feature'][tmp_node_type]= tmp_node_all_feature_pd.drop(columns=[tmp_node_type + '_UID']).values
            
            # 获取UID到index的对应关系，作为各节点的数值序号
            tmp_UID_to_Index_pd = tmp_node_all_feature_pd[[tmp_node_type + '_UID']].reset_index()
            tmp_node_UID_to_index_dict[tmp_node_type] = tmp_UID_to_Index_pd
            
            continue
        
        # 读取涉及的各类型节点表，并合并
        tmp_all_feature_pd_list = []
        for tmp_feature_type in Feature_Type_list:
            # 获取特征文件位置
            tmp_node_feature_file = tmp_all_output_data_time_range_feature_dir + tmp_node_type + '_' + tmp_feature_type + '.pkl'

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
        Result_Data_dict['Feature'][tmp_node_type] = tmp_node_all_feature_pd.drop(columns=[tmp_node_type + '_UID']).values
        
        # 获取UID到index的对应关系，作为各节点的数值序号
        tmp_UID_to_Index_pd = tmp_node_all_feature_pd[[tmp_node_type + '_UID']].reset_index()
        tmp_node_UID_to_index_dict[tmp_node_type] = tmp_UID_to_Index_pd
        
        # 存储拼接后的特征
        tmp_node_all_feature_pd.to_pickle(tmp_all_output_data_time_range_Model_Feature_dir + tmp_node_type + '_Feature.pkl')
    
    #######################################################################################################################
    # 查看是否已处理过标签表，若已处理过，则跳过处理，直接读取信息
    if os.path.exists(tmp_all_output_data_time_range_Model_dir + 'Target_Node_with_Index.pkl'):
        aim_UID_with_new_index_pd = pd.read_pickle(tmp_all_output_data_time_range_Model_dir + 'Target_Node_with_Index.pkl')
        
        Result_Data_dict['Label_Index'] = aim_UID_with_new_index_pd['index'].values
        
        if 'Label' in aim_UID_with_new_index_pd.columns:
            Result_Data_dict['Label'] = aim_UID_with_new_index_pd['Label'].values
    else:
        # 按新序号重新排列标签表中的标签序号
        aim_UID_with_new_index_pd = aim_UID_pd.merge(tmp_node_UID_to_index_dict[aim_node_type], on = aim_node_type + '_UID', 
                                      how = 'left').reset_index(drop = True)

        # 存储有标签的index号，以及对应的顺序
        aim_UID_with_new_index_pd.to_pickle(tmp_all_output_data_time_range_Model_dir + 'Target_Node_with_Index.pkl')

        # 保存标签信息到返回信息中
        Result_Data_dict['Label_Index'] = aim_UID_with_new_index_pd['index'].values
        
        # 确认全部index都不为nan
        if np.sum(np.isnan(Result_Data_dict['Label_Index'])) != 0:
            print('ERROR: Label_Index中nan值数为:', np.sum(np.isnan(Result_Data_dict['Label_Index'])))
        
        if 'Label' in aim_UID_with_new_index_pd.columns:
            Result_Data_dict['Label'] = aim_UID_with_new_index_pd['Label'].values
    
    #######################################################################################################################
    # 获取生成好的全部元路径
    tmp_meta_path_file_list = os.listdir(tmp_all_output_data_time_range_meta_path_dir)
    tmp_meta_path_file_list = [tmp_file_name for tmp_file_name in tmp_meta_path_file_list if '.pkl' in tmp_file_name]

    # 依次读取各元路径对应的数据
    for tmp_meta_path_file in tmp_meta_path_file_list:
        tmp_meta_path_name = tmp_meta_path_file.split('.pkl')[0]
        
        # 跳过不需要的元路径
        if tmp_meta_path_name in Meta_path_drop_list:
            print('跳过元路径:', tmp_meta_path_name)
            continue
        
        # 读取元路径对应的节点类型
        tmp_column_to_node_class_dict = np.load(tmp_all_output_data_time_range_meta_path_dir + tmp_meta_path_name
                                   + '-column_to_node_class.npy', allow_pickle= True).item()
        
        # 保留每条元路径上涉及到的全部的节点列，再根据节点类型转换index后保存结果
        tmp_all_aim_column_name_list = list(tmp_column_to_node_class_dict.keys())
        tmp_all_aim_node_column_name_list = [x for x in tmp_all_aim_column_name_list if 'Weight_' not in x]
        
        # 保存每列对应的节点类型
        Result_Data_dict['Adj_Node_Type'][tmp_meta_path_name] = [tmp_column_to_node_class_dict[tmp_column_name] for tmp_column_name in tmp_all_aim_node_column_name_list]
        
        # 查看是否已处理过对应邻接表，若已处理过，则跳过处理，直接读取信息
        if os.path.exists(tmp_all_output_data_time_range_Model_Adj_dir + tmp_meta_path_name + '.pkl'):
            print(tmp_meta_path_name + '对应邻接表已处理过，直接读取')
            tmp_meta_path_result_pandas = pd.read_pickle(tmp_all_output_data_time_range_Model_Adj_dir + tmp_meta_path_name + '.pkl')
            Result_Data_dict['Adj'][tmp_meta_path_name] = tmp_meta_path_result_pandas.values.T
            
            continue
        
        # 读取元路径信息
        tmp_meta_path_result_pandas = pd.read_pickle(tmp_all_output_data_time_range_meta_path_dir + tmp_meta_path_file)
        print('读取元路径:', tmp_meta_path_name)
        
        # 只保留节点列
        tmp_meta_path_result_pandas = tmp_meta_path_result_pandas[tmp_all_aim_node_column_name_list]
        
        # 依次对每列的index进行转换
        for tmp_column_name in tmp_all_aim_node_column_name_list:
            tmp_column_node_type = tmp_column_to_node_class_dict[tmp_column_name]
            
            # 获取该列对应的index
            tmp_meta_path_result_pandas = tmp_meta_path_result_pandas.merge(tmp_node_UID_to_index_dict[tmp_column_node_type], 
                                                       left_on = tmp_column_name,
                                                       right_on = tmp_column_node_type + '_UID', how = 'left')

            tmp_meta_path_result_pandas = tmp_meta_path_result_pandas.drop(columns=[tmp_column_name, tmp_column_node_type + '_UID'])

            tmp_meta_path_result_pandas = tmp_meta_path_result_pandas.rename(columns={'index' : tmp_column_name})
            
            # 确认该列中没有nan值
            if np.sum(np.isnan(tmp_meta_path_result_pandas[tmp_column_name].values)) != 0:
                print('ERROR: 该元路径首列中nan值数为:', np.sum(np.isnan(tmp_meta_path_result_pandas[tmp_column_name].values)))
        
        # 保存邻接表信息到返回信息中
        Result_Data_dict['Adj'][tmp_meta_path_name] = tmp_meta_path_result_pandas.values.T
        
        # 存储结果
        tmp_meta_path_result_pandas.to_pickle(tmp_all_output_data_time_range_Model_Adj_dir + tmp_meta_path_name + '.pkl')
        
    return Result_Data_dict