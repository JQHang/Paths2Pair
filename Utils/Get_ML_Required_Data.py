import numpy as np
import pandas as pd
import os
import json

from Utils.utils import mkdir

from pyspark.sql.types import *
from pyspark.sql.functions import broadcast

"""
作用：
    获取目标节点对应的特征
"""
def get_aim_node_result_column_feature(tmp_output_processed_aim_node_feature_file, aim_UID_pd, regenerate, Feature_Type_list, 
                          tmp_all_output_data_time_range_feature_dir, aim_node_type, sample_start, sample_end):
    # 如果还未生成，则生成
    if regenerate or not os.path.exists(tmp_output_processed_aim_node_feature_file):
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
        aim_node_feature_pd.to_pickle(tmp_output_processed_aim_node_feature_file)
        
    else:
        print('目标节点本身特征已存在')
        
        aim_node_feature_pd = pd.read_pickle(tmp_output_processed_aim_node_feature_file)
    
    if sample_start != None:
        aim_node_feature_pd = aim_node_feature_pd.iloc[sample_start:sample_end, :]
    
    return aim_node_feature_pd

"""
作用：
    获取目标元路径对应的特征
"""
def get_meta_path_result_column_feature(tmp_final_data_store_file, aim_UID_pd, regenerate, tmp_all_output_data_time_range_meta_path_dir, 
                           tmp_meta_path_file, tmp_column_to_node_class_dict, tmp_all_output_data_time_range_feature_dir, Feature_Type_list, 
                           Feature_Groupby_Type_list, aim_node_column_name, sample_start, sample_end):
    tmp_meta_path_name = tmp_meta_path_file.split('.pkl')[0]
    
    # 查看是否已完成生成
    if not regenerate and os.path.exists(tmp_final_data_store_file):
        # 读取已生成好的结果
        tmp_processed_feature_pd = pd.read_pickle(tmp_final_data_store_file)

        if sample_start != None:
            tmp_processed_feature_pd = tmp_processed_feature_pd.iloc[sample_start:sample_end, :]

        print('元路径' + tmp_meta_path_name + '已处理过，跳过，读取维度为', tmp_processed_feature_pd.shape)
        return tmp_processed_feature_pd

    # 读取元路径信息
    tmp_meta_path_result_pandas = pd.read_pickle(tmp_all_output_data_time_range_meta_path_dir + tmp_meta_path_file)
    print('读取元路径:', tmp_meta_path_name, tmp_meta_path_result_pandas.shape)

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

    print('读取完尾结点特征:', tmp_meta_path_name, tmp_related_UID_feature_pd.shape)

    # 对节点特征按目标列计算均值、最小、最大
    tmp_related_UID_feature_pd = tmp_related_UID_feature_pd.groupby(first_column_name).agg(Feature_Groupby_Type_list)

    print('处理完尾结点特征groupby结果:', tmp_meta_path_name, tmp_related_UID_feature_pd.shape)

    # 修正列名
    tmp_related_UID_feature_pd.columns = tmp_related_UID_feature_pd.columns.map(lambda x: '_'.join(filter(None, x)))

    # 将对应数据与标签表拼接，未拼接上的置0
    tmp_result_column_feature_pd = aim_UID_pd.merge(tmp_related_UID_feature_pd, left_on = aim_node_column_name, right_index = True,
                                    how = 'left')

    tmp_result_column_feature_pd = tmp_result_column_feature_pd.fillna(0)

    # 删去目标列
    tmp_result_column_feature_pd = tmp_result_column_feature_pd.drop(columns=[aim_node_column_name])

    # 给剩余列的列名加上元路径名称
    tmp_result_column_feature_pd.columns = tmp_meta_path_name + '___' + tmp_result_column_feature_pd.columns

    # 打印最终列数
    print('元路径' + tmp_meta_path_name + '处理后维度:', tmp_result_column_feature_pd.shape)

    # 保存结果
    tmp_result_column_feature_pd.to_pickle(tmp_final_data_store_file)
    
    # 如果有要求则只返回指定范围内的结果
    if sample_start != None:
        tmp_result_column_feature_pd = tmp_result_column_feature_pd.iloc[sample_start:sample_end, :]
    
    return tmp_result_column_feature_pd

"""
作用：
    获得能符合ML模型要求的pandas数据（包含标签列 + 特征列）。

输入：
    sample_source_str：目标节点来源的字符串描述（为了定位元路径、节点、特征的存储位置）
    time_range_str：目标节点对应的时间区间或点的字符串描述（也是为了定位元路径、节点、特征的存储位置）
    aim_ML_file_store_name:机器学习数据存储文件夹
    Meta_path_drop_list:不需要的元路径名称列表，会自动删去对应元路径
    Feature_Type_list:使用的特征类型（可以使用Raw、Norm、Std）
    
返回值：
    输出结果直接存储在对应文件夹中
"""
def get_ML_required_pandas_data(sample_source_str, time_range_str, aim_ML_file_store_name, aim_node_type, regenerate = False,
                      Meta_path_drop_list = [], Feature_Type_list = ['Norm'], 
                      Feature_Groupby_Type_list = ['mean', 'min', 'max', 'sum', 'count'], 
                      sample_start = None, sample_end = None):
    
    Processed_ML_Data_dict = {}
    
    tmp_all_output_data_base_dir = '../../../Data/'
    tmp_all_output_data_time_range_dir = ('../../../Data/' + sample_source_str + '/' +  time_range_str + '/')
    
    print('预处理' + tmp_all_output_data_time_range_dir + '文件夹下的相关文件')

    # 设置目标元路径存储文件夹
    tmp_all_output_data_time_range_meta_path_dir = tmp_all_output_data_time_range_dir + 'Meta_Path/'

    # 设置节点特征存储文件夹
    tmp_all_output_data_time_range_feature_dir = tmp_all_output_data_time_range_dir + 'Feature/'
    
    # 读取标签表
    label_pd = pd.read_pickle(tmp_all_output_data_time_range_dir + 'Target_Node.pkl')
    
    if 'Label' in label_pd.columns:
        Processed_ML_Data_dict['Label'] = label_pd['Label']
        print('Label Shape:', Processed_ML_Data_dict['Label'].shape)
        
    # 获取目标列列名
    aim_node_column_name = aim_node_type + '_UID'

    # 只保留其中非标签的列，作为目标节点列表
    aim_UID_pd = label_pd[[aim_node_column_name]].copy()
    print('目标点数目:', aim_UID_pd.shape[0])
    #######################################################################################################################
    # 设置ML所需的数据的存储位置
    tmp_all_output_data_time_range_ML_dir = tmp_all_output_data_time_range_dir + 'ML/'
    mkdir(tmp_all_output_data_time_range_ML_dir)
    
    # 查看是否已有对应文件
    if sample_start == None:
        if not regenerate and os.path.exists(tmp_all_output_data_time_range_ML_dir + aim_ML_file_store_name + '.pkl'):
            print('已处理过同名文件，直接返回')
            Processed_ML_Data_dict['Feature'] = pd.read_pickle(tmp_all_output_data_time_range_ML_dir + aim_ML_file_store_name + '.pkl')
            
            return Processed_ML_Data_dict
    else:
        if not regenerate and os.path.exists(tmp_all_output_data_time_range_ML_dir + aim_ML_file_store_name + '_' + str(sample_start) + '-' +
                                 str(sample_end) + '.pkl'):
            print('已处理过同名文件，直接返回')
            Processed_ML_Data_dict['Feature'] = pd.read_pickle(tmp_all_output_data_time_range_ML_dir + aim_ML_file_store_name + '_' + 
                                              str(sample_start) + '-' + str(sample_end) + '.pkl')
            
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
        
        # 读取元路径对应的节点类型
        tmp_column_to_node_class_dict = np.load(tmp_all_output_data_time_range_meta_path_dir + tmp_meta_path_name 
                                                + '-column_to_node_class.npy', allow_pickle= True).item()
        
        # 设置该元路径处理后结果的存储文件
        tmp_final_data_store_file = tmp_all_output_data_time_range_ML_dir + tmp_meta_path_name + '_Processed.pkl'
        
        # 获取结果
        tmp_result_column_feature_pd = get_meta_path_result_column_feature(tmp_final_data_store_file, aim_UID_pd, regenerate, 
                                                     tmp_all_output_data_time_range_meta_path_dir, tmp_meta_path_file, 
                                                     tmp_column_to_node_class_dict, tmp_all_output_data_time_range_feature_dir,
                                                     Feature_Type_list, Feature_Groupby_Type_list, 
                                                     aim_node_column_name, sample_start, sample_end)

        tmp_aim_ML_pandas_list.append(tmp_result_column_feature_pd)
    #######################################################################################################################
    # 目标节点本身特征存储位置
    tmp_output_processed_aim_node_feature_file = tmp_all_output_data_time_range_ML_dir + 'Aim_Node_Feature_Processed.pkl'
    
    aim_node_feature_pd = get_aim_node_result_column_feature(tmp_output_processed_aim_node_feature_file, aim_UID_pd, regenerate, Feature_Type_list, 
                                          tmp_all_output_data_time_range_feature_dir, aim_node_type, sample_start, sample_end)
    
    tmp_aim_ML_pandas_list.append(aim_node_feature_pd)
    #######################################################################################################################
    # 合并全部可用元路径结果
    tmp_aim_ML_pandas = pd.concat(tmp_aim_ML_pandas_list, axis = 1)
    
    # 保存结果
    if sample_start == None:
        tmp_aim_ML_pandas.to_pickle(tmp_all_output_data_time_range_ML_dir + aim_ML_file_store_name + '.pkl')
    else:
        tmp_aim_ML_pandas.to_pickle(tmp_all_output_data_time_range_ML_dir + aim_ML_file_store_name + '_' + str(sample_start) + '-' +
                           str(sample_end) + '.pkl')
        
    Processed_ML_Data_dict['Feature'] = tmp_aim_ML_pandas
    
    return Processed_ML_Data_dict

"""
作用：
    从线上表中读取符合ML模型要求的pandas数据（包含标签列 + 特征列）。

输入：
    aim_ML_file_store_name:机器学习数据存储文件夹
    Meta_path_drop_list:不需要的元路径名称列表，会自动删去对应元路径
    
返回值：
    Result_Data_dict
"""
def get_ML_required_pandas_data_from_online(Spark_Session, Aim_UID_info_dict, Feature_Data_From_Online_Config_dict, 
                              Feature_Data_From_Online_Time_Store_dir, aim_ML_file_store_name, regenerate, 
                              Meta_path_drop_list = [], Meta_path_Column_drop_dict = [], Preprocess_Type_List = ['Raw'], 
                              sample_start = None, sample_end = None):
    # 要输出的字典
    Result_Data_dict = {}
    Processed_Feature_List = []
    Processed_Feature_Summary_List = []
    
    ###########################################################################################################################
    # 读取目标点相关信息
    tmp_aim_entity_rdd = Aim_UID_info_dict['Data']
    Aim_Feature_Table_dt = Aim_UID_info_dict['Monthly_dt']
    Aim_Node_type = Aim_UID_info_dict['Node_Type']
    Aim_Node_UID_Name = Aim_Node_type + '_UID'
    
    # 只保留rdd中的目标节点列，去缩减特征表
    tmp_aim_node_UID_rdd = tmp_aim_entity_rdd.select([Aim_Node_UID_Name])
    
    # broadcast目标列，加快运算速度
    tmp_aim_node_UID_rdd_Broadcast = broadcast(tmp_aim_node_UID_rdd)
    
    # 保存目标点的顺序
    if 'Data_pd' in Aim_UID_info_dict:
        tmp_aim_entity_pd = Aim_UID_info_dict['Data_pd']
    else:
        tmp_aim_entity_pd = tmp_aim_entity_rdd.toPandas()

    # 如果有设定的范围，则只保留指定范围行内的标签
    if sample_start != None:
        print('只保留', sample_start, sample_end, '范围内的点和特征')
        tmp_aim_entity_pd = tmp_aim_entity_pd.iloc[sample_start:sample_end, :]

    tmp_aim_node_UID_order_pd = tmp_aim_entity_pd[[Aim_Node_UID_Name]].copy()
    
    # 如果有标签列，则保存标签列
    if 'Label' in tmp_aim_entity_pd.columns:
        Result_Data_dict['Label'] = tmp_aim_entity_pd['Label']

        print('Label Shape:', Result_Data_dict['Label'].shape)
            
    ###########################################################################################################################
    # 目标特征保存位置
    if sample_start == None:
        Aim_ML_Required_Raw_Feature_file = Feature_Data_From_Online_Time_Store_dir + aim_ML_file_store_name + '_Raw.pkl'
        Aim_ML_Required_Norm_Feature_file = Feature_Data_From_Online_Time_Store_dir + aim_ML_file_store_name + '_Norm.pkl'
        Aim_ML_Required_Std_Feature_file = Feature_Data_From_Online_Time_Store_dir + aim_ML_file_store_name + '_Std.pkl'
    else:
        Aim_ML_Required_Raw_Feature_file = (Feature_Data_From_Online_Time_Store_dir + aim_ML_file_store_name + '_' + 
                                str(sample_start) + '-' + str(sample_end) + '_Raw.pkl')
        Aim_ML_Required_Norm_Feature_file = (Feature_Data_From_Online_Time_Store_dir + aim_ML_file_store_name + '_' + 
                                 str(sample_start) + '-' + str(sample_end) + '_Norm.pkl')
        Aim_ML_Required_Std_Feature_file = (Feature_Data_From_Online_Time_Store_dir + aim_ML_file_store_name + '_' + 
                                str(sample_start) + '-' + str(sample_end) + '_Std.pkl')
    
    print('目标特征保存位置:', Aim_ML_Required_Raw_Feature_file)
    ###########################################################################################################################
    print('----------------------------------------------------------------------------')
    
    if regenerate or not os.path.exists(Aim_ML_Required_Raw_Feature_file):
        print('开始生成起始点本身特征')
        
        # 起始节点本身特征保存的位置
        Start_Node_Feature_file = Feature_Data_From_Online_Time_Store_dir + 'Start_Node.pkl'

        # 起始节点涉及到的表的统计信息
        Start_Node_Feature_Summary_file = Feature_Data_From_Online_Time_Store_dir + 'Start_Node_Summary.pkl'

        # 只保留rdd中的目标节点列，去拼接特征
        tmp_aim_node_feature_rdd = tmp_aim_entity_rdd.select([Aim_Node_UID_Name])
        
        if regenerate or not os.path.exists(Start_Node_Feature_file):
            # 记录各个特征表涉及的全部特征
            tmp_all_useful_feature_cols_list = []

            # 记录各个特征表涉及的全部特征的注释
            tmp_all_useful_feature_cols_comments_list = []
            
            # 记录各个特征表的统计信息
            tmp_feature_table_summary_pd_list = []

            for tmp_feature_table_Info_dict in Feature_Data_From_Online_Config_dict['Start_Node_Feature_Table_List']:
                tmp_feature_table_name = tmp_feature_table_Info_dict['Table_Name']
                tmp_aim_column_name = tmp_feature_table_Info_dict['UID']
                
                print('读取特征表:', tmp_feature_table_name)
                
                tmp_sql_command = """
                            SELECT
                                *
                            FROM
                                """ + tmp_feature_table_name + """
                            WHERE 
                                dt = '""" + Aim_Feature_Table_dt + """'
                            """

                tmp_feature_table_rdd = Spark_Session.sql(tmp_sql_command)

                # 确保添加特征目标列的列名为tmp_node_class + '_UID'
                tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(tmp_aim_column_name, Aim_Node_type + '_UID')

                # 通过persist保留计算结果
                tmp_feature_table_rdd = tmp_feature_table_rdd.persist()

                tmp_feature_table_rdd_raw_count = tmp_feature_table_rdd.count()

                if tmp_feature_table_rdd_raw_count == 0:
                    print('Error: 特征表', tmp_feature_table_name, '为空，得及时处理')
                else:
                    # 对特征表去重（理论上不需要，但防止有不符合规范的表）
                    tmp_feature_table_rdd = tmp_feature_table_rdd.dropDuplicates([Aim_Node_type + '_UID'])

                    # 通过persist保留计算结果
                    tmp_feature_table_rdd = tmp_feature_table_rdd.persist()

                    tmp_feature_table_rdd_count = tmp_feature_table_rdd.count()

                    if tmp_feature_table_rdd_raw_count != tmp_feature_table_rdd_count:
                        print('Error: 特征表', tmp_feature_table_name, '内部有重复UID，得及时修改, 目前先保留第一条信息')

                # 获取特征表的格式信息
                tmp_feature_table_rdd_json = json.loads(tmp_feature_table_rdd.schema.json())['fields']

                # 记录有效的特征列名
                tmp_useful_feature_cols_list = []

                # 记录有效的特征注释名
                tmp_useful_feature_cols_comments_list = []

                # 只保留其中有效的列（entity_id加数值类型的列，*待优化，数据格式标准化后改为entity_id及其之后的列）
                for tmp_col_info in tmp_feature_table_rdd_json:
                    col = tmp_col_info['name']
                    col_type = tmp_col_info['type']

                    if col == (Aim_Node_type + '_UID'):
                        continue

                    if 'entity_id' in col:
                        continue

                    if col_type in ['int', 'integer', 'float', 'bigint','double', 'long']:
                        tmp_transferred_column_name = (col + '___' + tmp_feature_table_name.split('.')[-1])
                        if 'comment' in tmp_col_info['metadata']:
                            col_comment = (tmp_col_info['metadata']['comment'] + '___' + tmp_feature_table_name.split('.')[-1])
                        else:
                            col_comment = tmp_transferred_column_name

                        tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(col, tmp_transferred_column_name)

                        tmp_useful_feature_cols_list.append(tmp_transferred_column_name)

                        tmp_useful_feature_cols_comments_list.append(col_comment)
                    elif col_type != 'string':
                        print('-----------------------------------------------------------')
                        print('WARNING:stange_type:', col, col_type)
                        print('-----------------------------------------------------------')

                tmp_feature_table_rdd = tmp_feature_table_rdd.select([Aim_Node_type + '_UID'] + tmp_useful_feature_cols_list)

                # 保存列名信息
                tmp_all_useful_feature_cols_list.extend(tmp_useful_feature_cols_list)
                tmp_all_useful_feature_cols_comments_list.extend(tmp_useful_feature_cols_comments_list)

                print('特征表'+ tmp_feature_table_name + '添加特征数:', len(tmp_useful_feature_cols_list))

                # 通过persist保留计算结果
                tmp_feature_table_rdd = tmp_feature_table_rdd.persist()

                # 只保留目标点的数据(通过broadcast加快运算速度)
                tmp_sub_feature_table_rdd = tmp_feature_table_rdd.join(tmp_aim_node_UID_rdd_Broadcast, Aim_Node_type + '_UID', 'inner')
                
                # join两个表
                tmp_aim_node_feature_rdd = tmp_aim_node_feature_rdd.join(tmp_sub_feature_table_rdd, Aim_Node_type + '_UID', 'left')
                
                # 计算除了UID列的min, max, mean, std，并转化为pandas
                tmp_feature_table_summary_pd = tmp_feature_table_rdd.select(tmp_useful_feature_cols_list).summary("min", "max", 
                                                                                 "mean", "stddev").toPandas()
                
                # 查看是否有无效列(特征都为同一值)，及时提醒
                tmp_summary_min = tmp_feature_table_summary_pd[tmp_feature_table_summary_pd['summary'] == 'min'].values[0]
                tmp_summary_max = tmp_feature_table_summary_pd[tmp_feature_table_summary_pd['summary'] == 'max'].values[0]

                tmp_problem_columns = np.array(tmp_feature_table_summary_pd.columns)[tmp_summary_min == tmp_summary_max]

                if tmp_problem_columns.shape[0] > 0:
                    print('ERROR: 存在特征列全部行都是一个值，具体情况如下，得及时修改')
                    print(dict(tmp_feature_table_summary_pd[tmp_problem_columns].iloc[0]))

                # 删除summary列后记录结果
                tmp_feature_table_summary_pd_list.append(tmp_feature_table_summary_pd.set_index('summary'))

            print(Aim_Node_type + '目标节点涉及的全部特征数:', len(tmp_all_useful_feature_cols_list))

            tmp_aim_node_feature_pandas = tmp_aim_node_feature_rdd.toPandas()
            
            tmp_aim_node_feature_pandas = tmp_aim_node_feature_pandas.fillna(0)
            
            tmp_aim_node_feature_pandas.to_pickle(Start_Node_Feature_file)

            # 合并各特征表的统计结果
            tmp_feature_table_summary_pd = pd.concat(tmp_feature_table_summary_pd_list, axis = 1)

            # 保存各列的统计结果
            tmp_feature_table_summary_pd.to_pickle(Start_Node_Feature_Summary_file)

        else:
            print('起始点特征表已存在，直接读取')

            tmp_aim_node_feature_pandas = pd.read_pickle(Start_Node_Feature_file)

            tmp_feature_table_summary_pd = pd.read_pickle(Start_Node_Feature_Summary_file)
            
            print(Aim_Node_type + '目标节点涉及的全部特征数:', tmp_feature_table_summary_pd.shape[1])
            
        # 根据标签的顺序修正pd表中的特征的顺序，并只保留需要范围内的特征
        tmp_aim_node_feature_pandas = tmp_aim_node_UID_order_pd.merge(tmp_aim_node_feature_pandas, how = 'left', on = Aim_Node_type + '_UID')
        
        Processed_Feature_List.append(tmp_aim_node_feature_pandas.drop(columns=[Aim_Node_type + '_UID']))
        Processed_Feature_Summary_List.append(tmp_feature_table_summary_pd)
        ###########################################################################################################################
        print('----------------------------------------------------------------------------')
        print('处理元路径对应的特征')
        
        # 获取配置中的全部元路径
        tmp_meta_path_name_list = list(Feature_Data_From_Online_Config_dict['Meta_Path_Feature_Table_List'].keys())

        # 保证处理顺序一致
        tmp_meta_path_name_list.sort()
        
        print('要处理的全部元路径及顺序为:', tmp_meta_path_name_list)
        
        # 依次读取各元路径对应的数据
        for tmp_meta_path_name in tmp_meta_path_name_list:
            # 跳过不需要的元路径
            if tmp_meta_path_name in Meta_path_drop_list:
                print('跳过元路径:', tmp_meta_path_name)
                continue

            print('处理元路径:', tmp_meta_path_name)

            # 获得对应元路径的配置信息
            tmp_meta_path_info = Feature_Data_From_Online_Config_dict['Meta_Path_Feature_Table_List'][tmp_meta_path_name]
            
            # 依次处理元路径中各列的信息
            for tmp_column_i in range(len(tmp_meta_path_info)):
                # 对应列节点类型
                tmp_column_node_class = tmp_meta_path_info[tmp_column_i]["Node_class"]
                
                # 跳过不需要的元路径列
                if tmp_meta_path_name in Meta_path_Column_drop_dict and tmp_column_i in Meta_path_Column_drop_dict[tmp_meta_path_name]:
                    print('跳过元路径:', tmp_meta_path_name, '中的第', tmp_column_i, '列')
                    continue
                
                if tmp_column_node_class == 'Weight':
                    print('跳过权重列')
                    continue
                
                print('处理元路径:', tmp_meta_path_name, '中的第', tmp_column_i, '列')
            
                # 该元路径对应的特征保存的位置
                tmp_meta_path_Feature_file = (Feature_Data_From_Online_Time_Store_dir + tmp_meta_path_name + '_' + str(tmp_column_i) 
                                    + '_' + tmp_column_node_class + '.pkl')

                # 该元路径对应的特征的统计信息的保存的位置
                tmp_meta_path_Feature_summary_file = (Feature_Data_From_Online_Time_Store_dir + tmp_meta_path_name + '_' + str(tmp_column_i) 
                                          + '_' + tmp_column_node_class +  '_Summary.pkl')

                # 只保留rdd中的目标节点列，去拼接特征
                tmp_meta_path_node_feature_rdd = tmp_aim_entity_rdd.select([Aim_Node_UID_Name])

                # 如果还没生成特征，就进行生成
                if regenerate or not os.path.exists(tmp_meta_path_Feature_file):
                    # 记录各个特征表的统计信息
                    tmp_feature_table_summary_pd_list = []

                    # 机器学习模型只用处理最后一列对应的特征表即可
                    for tmp_feature_table_name in tmp_meta_path_info[tmp_column_i]["Feature_Table_List"]:
                        print('处理特征表', tmp_feature_table_name)
                        
                        tmp_aim_column_name = 'Start_Column'

                        tmp_sql_command = """
                                    SELECT
                                        *
                                    FROM
                                        """ + tmp_feature_table_name + """
                                    WHERE 
                                        dt = '""" + Aim_Feature_Table_dt + """'
                                    """

                        tmp_feature_table_rdd = Spark_Session.sql(tmp_sql_command)

                        # 确保添加特征目标列的列名为tmp_node_class + '_UID'
                        tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(tmp_aim_column_name, Aim_Node_type + '_UID')

                        # 删去dt列
                        tmp_feature_table_rdd = tmp_feature_table_rdd.drop('dt')

                        # 通过persist保留计算结果
                        tmp_feature_table_rdd = tmp_feature_table_rdd.persist()
                        
                        # 只保留目标点的数据(通过broadcast加快运算速度)
                        tmp_sub_feature_table_rdd = tmp_feature_table_rdd.join(tmp_aim_node_UID_rdd_Broadcast, Aim_Node_type + '_UID', 'inner')
                        
                        # 去重（因为可能由于分批传输数据间的中断造成有重复特征）
                        tmp_sub_feature_table_rdd = tmp_sub_feature_table_rdd.dropDuplicates([Aim_Node_type + '_UID'])
                        
                        # join两个表(因为已经是处理后的表，所以不用再检测是否是数值类型的特征)
                        tmp_meta_path_node_feature_rdd = tmp_meta_path_node_feature_rdd.join(tmp_sub_feature_table_rdd, Aim_Node_type + '_UID', 'left')

                        # 计算除了UID列的min, max, mean, std，并转化为pandas
                        tmp_feature_table_summary_pd = tmp_feature_table_rdd.drop(Aim_Node_type + '_UID').summary("min", "max", "mean", "stddev").toPandas()

                        # 查看是否有无效列(特征都为同一值)，及时提醒
                        tmp_summary_min = tmp_feature_table_summary_pd[tmp_feature_table_summary_pd['summary'] == 'min'].values[0]
                        tmp_summary_max = tmp_feature_table_summary_pd[tmp_feature_table_summary_pd['summary'] == 'max'].values[0]

                        tmp_problem_columns = np.array(tmp_feature_table_summary_pd.columns)[tmp_summary_min == tmp_summary_max]

                        if tmp_problem_columns.shape[0] > 0:
                            print('ERROR: 存在特征列全部行都是一个值，具体情况如下，得及时修改')
                            print(dict(tmp_feature_table_summary_pd[tmp_problem_columns].iloc[0]))

                        # 删除summary列后记录结果
                        tmp_feature_table_summary_pd_list.append(tmp_feature_table_summary_pd.set_index('summary'))

                    tmp_meta_path_node_feature_pd = tmp_meta_path_node_feature_rdd.toPandas()
                    
                    # 给特征列名都加上元路径的信息
                    tmp_new_column_name_list = []
                    for tmp_old_column_name in tmp_meta_path_node_feature_pd.columns:
                        if tmp_old_column_name == Aim_Node_type + '_UID':
                            tmp_new_column_name_list.append(tmp_old_column_name)
                        else:
                            tmp_new_column_name_list.append(tmp_old_column_name + '___' + tmp_meta_path_name + '___' + 
                                                  str(tmp_column_i))
                    tmp_meta_path_node_feature_pd.columns = tmp_new_column_name_list
                    
                    # 保存特征表
                    tmp_meta_path_node_feature_pd.to_pickle(tmp_meta_path_Feature_file)

                    # 合并各特征表的统计结果
                    tmp_feature_table_summary_pd = pd.concat(tmp_feature_table_summary_pd_list, axis = 1)
                    
                    # 给统计表中的特征列名都加上元路径的信息
                    tmp_new_column_name_list = []
                    for tmp_old_column_name in tmp_feature_table_summary_pd.columns:
                        if tmp_old_column_name == Aim_Node_type + '_UID':
                            tmp_new_column_name_list.append(tmp_old_column_name)
                        else:
                            tmp_new_column_name_list.append(tmp_old_column_name + '___' + tmp_meta_path_name + '___' + 
                                                  str(tmp_column_i))
                    tmp_feature_table_summary_pd.columns = tmp_new_column_name_list
                    
                    # 保存各列的统计结果
                    tmp_feature_table_summary_pd.to_pickle(tmp_meta_path_Feature_summary_file)

                else:
                    print('该元路径的特征表已存在，直接读取')

                    tmp_meta_path_node_feature_pd = pd.read_pickle(tmp_meta_path_Feature_file)

                    tmp_feature_table_summary_pd = pd.read_pickle(tmp_meta_path_Feature_summary_file)
                
                print('特征维度为:', tmp_meta_path_node_feature_pd.shape)
                
                # 根据标签的顺序修正pd表中的特征的顺序，并只保留需要范围内的特征
                tmp_meta_path_node_feature_pd = tmp_aim_node_UID_order_pd.merge(tmp_meta_path_node_feature_pd, how = 'left', 
                                                           on = Aim_Node_type + '_UID')
            
                Processed_Feature_List.append(tmp_meta_path_node_feature_pd.drop(columns=[Aim_Node_type + '_UID']))
                Processed_Feature_Summary_List.append(tmp_feature_table_summary_pd)
                
            print('----------------------------------------------------------------------------')
        ###########################################################################################################################
        print('开始进行归一化及标准化')
        
        # 合并各元路径对应的数据为一个表
        tmp_aim_ML_pandas = pd.concat(Processed_Feature_List, axis = 1)
        tmp_aim_ML_pandas = tmp_aim_ML_pandas.fillna(0) 
        
        # 获得各列的统计信息
        tmp_aim_ML_pandas_summary = pd.concat(Processed_Feature_Summary_List, axis = 1)

        # 确认各列列名一致，否则报错
        if list(tmp_aim_ML_pandas.columns) != list(tmp_aim_ML_pandas_summary.columns):
            print('Error: 特征列和总结列列名不一致')
            return

        # 对获得的数据进行预处理（即归一化和标准化）
        tmp_pd_min = tmp_aim_ML_pandas_summary.loc['min'].astype('float')
        tmp_pd_max = tmp_aim_ML_pandas_summary.loc['max'].astype('float')
        tmp_pd_mean = tmp_aim_ML_pandas_summary.loc['max'].astype('float')
        tmp_pd_std = tmp_aim_ML_pandas_summary.loc['max'].astype('float')

        # 进行归一化和标准化
        tmp_aim_ML_pandas_norm = ((tmp_aim_ML_pandas - tmp_pd_min) / (tmp_pd_max - tmp_pd_min))
        tmp_aim_ML_pandas_norm = tmp_aim_ML_pandas_norm.fillna(0) 

        tmp_aim_ML_pandas_std = ((tmp_aim_ML_pandas - tmp_pd_mean) / tmp_pd_std)
        tmp_aim_ML_pandas_std = tmp_aim_ML_pandas_std.fillna(0)

        ###########################################################################################################################
        # 保存结果
        tmp_aim_ML_pandas.to_pickle(Aim_ML_Required_Raw_Feature_file)
        tmp_aim_ML_pandas_norm.to_pickle(Aim_ML_Required_Norm_Feature_file)
        tmp_aim_ML_pandas_std.to_pickle(Aim_ML_Required_Std_Feature_file)

        # 保存需要的格式的数据
        All_Output_Feature_list = []
        if 'Raw' in Preprocess_Type_List:
            All_Output_Feature_list.append(tmp_aim_ML_pandas)
        if 'Norm' in Preprocess_Type_List:
            All_Output_Feature_list.append(tmp_aim_ML_pandas_norm)
        if 'Std' in Preprocess_Type_List:
            All_Output_Feature_list.append(tmp_aim_ML_pandas_std)
            
    else:
        print('目标特征数据已存在，直接读取')
        
        # 读取需要的格式的数据
        All_Output_Feature_list = []
        if 'Raw' in Preprocess_Type_List:
            All_Output_Feature_list.append(pd.read_pickle(Aim_ML_Required_Raw_Feature_file))
        if 'Norm' in Preprocess_Type_List:
            All_Output_Feature_list.append(pd.read_pickle(Aim_ML_Required_Norm_Feature_file))
        if 'Std' in Preprocess_Type_List:
            All_Output_Feature_list.append(pd.read_pickle(Aim_ML_Required_Std_Feature_file))
    
    Result_Data_dict['Feature'] = pd.concat(All_Output_Feature_list, axis = 1)
    
    print('完成数据生成')
    print('----------------------------------------------------------------------------')
    
    return Result_Data_dict