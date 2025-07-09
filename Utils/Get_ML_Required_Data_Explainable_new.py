import numpy as np
import pandas as pd
import os

from kg_lib.utils import mkdir

"""
作用：
    获得能符合ML模型要求的pandas数据（包含标签列 + 特征列）。

输入：
    sample_source_str：目标节点来源的字符串描述（为了定位元路径、节点、特征的存储位置）
    time_range_str：目标节点对应的时间区间或点的字符串描述（也是为了定位元路径、节点、特征的存储位置）
    aim_ML_file_store_name:机器学习数据存储文件夹
    Meta_path_drop_list:不需要的元路径名称列表，会自动删去对应元路径
    Feature_Type_list:使用的特征类型（可以使用Raw、Norm、Std）
更新：
    针对于当前拼接特征并进行预测的代码进行修改，对于设定的batch，将特征数据分批次进行保存，文件使用int
    类型数值序号来进行命名。对于已经保存的结果而言，我们是采用读取具体的文件并计算行数，然后类加行数，最
    终得到我们当前要接力去进行特征添加的位置。接着这个位置再继续向后执行，并进行结果的保存。

返回值：
    输出结果直接存储在对应文件夹中
"""
def get_ML_required_pandas_data(index,aim_UID_pd,sample_source_str, time_range_str, aim_ML_file_store_name, aim_node_type,
                      Meta_path_drop_list = [], Feature_Type_list = ['Raw']):
    '''
    :param index: 
    :param sample_source_str:
    :param time_range_str:
    :param aim_ML_file_store_name:
    :param aim_node_type:
    :param Meta_path_drop_list:
    :param Feature_Type_list:
    :param batch:
    :return:
    '''
    Processed_ML_Data_dict = {}
    
    tmp_all_output_data_base_dir = '../Data/'
    tmp_all_output_data_time_range_dir = ('../Data/' + sample_source_str + '/' +  time_range_str + '/')
    
    print('预处理' + tmp_all_output_data_time_range_dir + '文件夹下的相关文件')

    # 设置目标元路径存储文件夹
    tmp_all_output_data_time_range_meta_path_dir = tmp_all_output_data_time_range_dir + 'Meta_Path/'

    # 设置节点特征存储文件夹
    tmp_all_output_data_time_range_feature_dir = tmp_all_output_data_time_range_dir + 'Feature/'
    
    # 读取标签表
    # label_pd = pd.read_pickle(tmp_all_output_data_time_range_dir + 'Target_Node.pkl')
    # if 'Label' in label_pd.columns:
    #     Processed_ML_Data_dict['Label'] = label_pd['Label']
    
    # 获取目标列列名
    aim_node_column_name = aim_node_type + '_UID'

    # 只保留其中非标签的列，作为目标节点列表
    aim_UID_pd = aim_UID_pd[[aim_node_column_name]].copy()
    
    #######################################################################################################################
    # 设置ML所需的数据的存储位置
    tmp_all_output_data_time_range_ML_dir = tmp_all_output_data_time_range_dir + 'ML_Raw/'
    mkdir(tmp_all_output_data_time_range_ML_dir)
    
    # 查看是否已有对应文件
    if os.path.exists(tmp_all_output_data_time_range_ML_dir + aim_ML_file_store_name +'_'+str(index)+'.pkl'):
        print('已处理过同名文件，直接返回')
        Processed_ML_Data_dict['Feature'] = pd.read_pickle(tmp_all_output_data_time_range_ML_dir + aim_ML_file_store_name + '_'+str(index)+'.pkl')
        return Processed_ML_Data_dict
    #######################################################################################################################
    # 获取生成好的全部元路径
    tmp_meta_path_file_list = os.listdir(tmp_all_output_data_time_range_meta_path_dir)
    tmp_meta_path_file_list = [tmp_file_name for tmp_file_name in tmp_meta_path_file_list if '.pkl' in tmp_file_name]
    
    # 保证处理顺序一致
    tmp_meta_path_file_list.sort()
    
    # 全部可用pandas
    tmp_aim_ML_pandas_list = []
    
    # 依次读取各元路径对应的数据
    for tmp_meta_path_file in tmp_meta_path_file_list:
        tmp_meta_path_name = tmp_meta_path_file.split('.pkl')[0]

        # 跳过不需要的元路径
        if tmp_meta_path_name in Meta_path_drop_list:
            print('跳过元路径:', tmp_meta_path_name)
            continue
        
        print('处理元路径:', tmp_meta_path_name)
        
        # *******************
        # 此处最好跳过_all_nodes_and_edges相关元路径，只保留最简短的元路径，但因为我还没生成对应的文件故先不跳过
        
        # 设置该元路径处理后结果的存储文件夹
        tmp_output_processed_meta_path_dir = tmp_all_output_data_time_range_ML_dir + tmp_meta_path_name + '_Processed/'
        mkdir(tmp_output_processed_meta_path_dir)
        
        # 读取元路径对应的节点类型
        tmp_column_to_node_class_dict = np.load(tmp_all_output_data_time_range_meta_path_dir + tmp_meta_path_name 
                                                + '-column_to_node_class.npy', allow_pickle= True).item()
        
        # 查看是否已完成生成
        tmp_final_data_store_file = (tmp_output_processed_meta_path_dir + '_'+str(index)+'output.pkl')
        if os.path.exists(tmp_final_data_store_file):
            print('元路径' + tmp_meta_path_name + '已处理过，跳过')
            
            # 读取已生成好的结果
            tmp_processed_feature_pd = pd.read_pickle(tmp_final_data_store_file)
            tmp_aim_ML_pandas_list.append(tmp_processed_feature_pd)
            
            continue
        
        # 读取元路径信息
        tmp_meta_path_result_pandas = pd.read_pickle(tmp_all_output_data_time_range_meta_path_dir + tmp_meta_path_file)
        print('读取元路径:', tmp_meta_path_name)
        
        # 只保留第一列和当前列
        first_column_name = list(tmp_column_to_node_class_dict.keys())[0]
        last_column_name = list(tmp_column_to_node_class_dict.keys())[-1]
        tmp_related_UID_feature_pd = tmp_meta_path_result_pandas[[first_column_name, last_column_name]].copy()

        # 获取节点类型
        tmp_node_type = tmp_column_to_node_class_dict[last_column_name]

        # 根据需求，依次读取各类型的特征文件
        for tmp_feature_type in Feature_Type_list:
            # 获取特征文件位置
            tmp_node_feature_file = tmp_all_output_data_time_range_feature_dir + '0/'+ tmp_node_type + '_' + tmp_feature_type + '.pkl'

            # 读取特征文件
            tmp_node_feature_pd = pd.read_pickle(tmp_node_feature_file)
            
            # 去重(理论上用不到，特征表阶段应该已经去过重了，但以防万一)
            tmp_node_feature_pd = tmp_node_feature_pd.drop_duplicates([tmp_node_type + '_UID'], ignore_index = True)
            
            # 修正特征文件目标列列名
            tmp_node_feature_pd = tmp_node_feature_pd.rename(columns={tmp_node_type + '_UID': last_column_name})

            # 修正列名(除了目标列，都加上后缀)
            tmp_node_feature_pd.columns = tmp_node_feature_pd.columns.map(lambda x: x + '_' + tmp_feature_type if x != last_column_name else x)

            # 和目标节点表拼接
            tmp_related_UID_feature_pd = tmp_related_UID_feature_pd.merge(tmp_node_feature_pd, on = last_column_name, how = 'left')

        # 删去特征目标列
        tmp_related_UID_feature_pd = tmp_related_UID_feature_pd.drop(columns=[last_column_name])

        # 对节点特征按目标列计算均值、最小、最大
        tmp_related_UID_feature_pd = tmp_related_UID_feature_pd.groupby(first_column_name).agg(['mean', 'min', 'max'])

        # 修正列名
        tmp_related_UID_feature_pd.columns = tmp_related_UID_feature_pd.columns.map(lambda x: '_'.join(filter(None, x)))

        # 将index变成列(此时index为first_column_name)
        tmp_related_UID_feature_pd = tmp_related_UID_feature_pd.reset_index()

        # 保证第一列列名和目标列一致
        tmp_related_UID_feature_pd = tmp_related_UID_feature_pd.rename(columns={first_column_name: aim_node_column_name})

        # 将对应数据与标签表拼接，未拼接上的置0
        tmp_result_column_feature_pd = aim_UID_pd.merge(tmp_related_UID_feature_pd, on = aim_node_column_name, how = 'left')

        tmp_result_column_feature_pd = tmp_result_column_feature_pd.fillna(0)

        # 删去目标列
        tmp_result_column_feature_pd = tmp_result_column_feature_pd.drop(columns=[aim_node_column_name])
        
        # 给剩余列的列名加上元路径名称
        tmp_result_column_feature_pd.columns = tmp_meta_path_name + '___' + tmp_result_column_feature_pd.columns
        
        # 打印最终列数
        print('元路径' + tmp_meta_path_name + '处理后维度:', tmp_result_column_feature_pd.shape)

        # 保存结果
        tmp_result_column_feature_pd.to_pickle(tmp_output_processed_meta_path_dir +'_'+str(index)+ 'output.pkl')

        tmp_aim_ML_pandas_list.append(tmp_result_column_feature_pd)
    
    #######################################################################################################################
    # 目标节点本身特征存储位置
    tmp_output_processed_aim_node_feature_dir = tmp_all_output_data_time_range_ML_dir + 'Aim_Node_Feature_Processed/'
    mkdir(tmp_output_processed_aim_node_feature_dir)
    
    # 如果还未生成，则生成
    if not os.path.exists(tmp_output_processed_aim_node_feature_dir +'_'+str(index)+ 'output.pkl'):
        # 获取目标列本身的相关特征
        aim_node_feature_pd = aim_UID_pd.copy()

        for tmp_feature_type in Feature_Type_list:
            # 获取特征文件位置
            tmp_node_feature_file = tmp_all_output_data_time_range_feature_dir + '0/' + aim_node_type + '_' + tmp_feature_type + '.pkl'

            # 读取特征文件
            tmp_node_feature_pd = pd.read_pickle(tmp_node_feature_file)
            
            # 去重(理论上用不到，特征表阶段应该已经去过重了，但以防万一)
            tmp_node_feature_pd = tmp_node_feature_pd.drop_duplicates([aim_node_type + '_UID'], ignore_index = True)
            
            # 修正列名(除了目标列，都加上后缀)
            tmp_node_feature_pd.columns = tmp_node_feature_pd.columns.map(lambda x: x + '_' + tmp_feature_type if x != (aim_node_type + '_UID') else x)

            # 和目标节点表拼接(aim_node_column_name == aim_node_type + '_UID')
            aim_node_feature_pd = aim_node_feature_pd.merge(tmp_node_feature_pd, on = aim_node_type + '_UID', how = 'left')
        
        # 未拼接上的置0
        aim_node_feature_pd = aim_node_feature_pd.fillna(0)
        
        # 删去目标列
        aim_node_feature_pd = aim_node_feature_pd.drop(columns=[aim_node_type + '_UID'])
        
        # 给剩余列的列名加上特征来源名称
        aim_node_feature_pd.columns = 'Aim_Node_Feature___' + aim_node_feature_pd.columns
        
        # 打印最终列数
        print('目标节点特征处理后维度:', aim_node_feature_pd.shape)

        # 保存结果
        aim_node_feature_pd.to_pickle(tmp_output_processed_aim_node_feature_dir +'_'+str(index)+ 'output.pkl')
        
        tmp_aim_ML_pandas_list.append(aim_node_feature_pd)
    else:
        print('目标节点本身特征已存在')
        
        tmp_aim_ML_pandas_list.append(pd.read_pickle(tmp_output_processed_aim_node_feature_dir +'_'+str(index)+ 'output.pkl'))
    #######################################################################################################################
    # 合并全部可用元路径结果
    tmp_aim_ML_pandas = pd.concat(tmp_aim_ML_pandas_list, axis = 1)
    
    # 保存结果
    tmp_aim_ML_pandas.to_pickle(tmp_all_output_data_time_range_ML_dir + aim_ML_file_store_name +'_'+str(index)+ '.pkl')
    
    Processed_ML_Data_dict['Feature'] = tmp_aim_ML_pandas
    
    return Processed_ML_Data_dict