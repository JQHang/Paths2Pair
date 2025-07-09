from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.types import *

from Utils.Pyspark_utils import sample_random_n_samples_for_samll_rdd
from Utils.Pyspark_utils import sample_top_n_groupby_samples_for_samll_rdd
from Utils.Pyspark_utils import sample_random_n_groupby_samples_for_samll_rdd

import pandas as pd
import numpy as np
import os

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from Utils.utils import mkdir

"""
作用：
    为涉及到的全部节点拼接对应的全量特征，并将原始值、归一化后的值以及标准化后的值存储到指定文件夹下（运算前会先检测相关文件是否存在，若已存在，则直接跳过）

输入：
    Spark_Session：pyspark接口
    tmp_aim_entity_info_dict：目标节点的相关信息（主要使用其中的目标节点的对应日期去取指定时间的特征）
    Feature_Dataset_Config_dict：特征表配置文件
    tmp_output_node_dir：涉及到的全部节点存储文件夹
    tmp_output_node_feature_dir: 节点特征输出文件夹
    tmp_feature_month_range:获取节点过去几个月的特征
    
返回值：
    tmp_all_nodes_features_used_in_meta_path_dict：字典形式的元路径包含的各种节点对应的特征
    
"""
def get_node_related_features(Spark_Session, tmp_aim_entity_info_dict, Feature_Dataset_Config_dict, 
                    tmp_output_node_dir, tmp_output_node_feature_dir, tmp_feature_month_range = 1):
    # 获取目标时间
    tmp_aim_table_dt = tmp_aim_entity_info_dict['Monthly_dt']
    tmp_aim_table_dt_datetime = datetime.strptime(tmp_aim_table_dt, '%Y-%m-%d')
    
    # 读取node文件夹下的全部文件
    node_file_list = os.listdir(tmp_output_node_dir)
    
    # 只保留其中的pkl文件
    node_file_list = [tmp_file_name for tmp_file_name in node_file_list if '.pkl' in tmp_file_name]
    print('全部的节点表文件:', node_file_list)
    
    # 依次处理各个节点
    for tmp_file_name in node_file_list:
        tmp_node_class = tmp_file_name.split('.pkl')[0]
        print('处理节点:', tmp_node_class)

        # 读取节点文件
        tmp_aim_node_UID_pandas = pd.read_pickle(tmp_output_node_dir + tmp_file_name)
        
        # 依次处理各个时间的结果
        for tmp_past_month_num in range(tmp_feature_month_range):
            tmp_past_month_datetime = tmp_aim_table_dt_datetime - relativedelta(months = tmp_past_month_num)
            tmp_past_month_datetime = datetime(tmp_past_month_datetime.year, tmp_past_month_datetime.month, 1)

            tmp_current_table_month_dt = tmp_past_month_datetime.strftime("%Y-%m-%d")

            tmp_monthly_output_node_feature_dir = tmp_output_node_feature_dir + str(tmp_past_month_num) + '/'
            mkdir(tmp_monthly_output_node_feature_dir)

            # 设定对应类型节点特征的文件名
            tmp_output_data_node_feature_file = (tmp_monthly_output_node_feature_dir + tmp_node_class + '_Raw.pkl')

            ##############################################################################################
            # 查看是否已有相关文件
            if os.path.exists(tmp_output_data_node_feature_file):
                # 如果之前已有对应特征表，则先验证行数是否一致
                tmp_past_node_feature_pd = pd.read_pickle(tmp_output_data_node_feature_file)

                # 如果行数相同，则跳过
                if tmp_past_node_feature_pd.shape[0] == tmp_aim_node_UID_pandas.shape[0]:
                    print('-----------------------------------------------------------')
                    print('WARNING: 全部' + tmp_node_class + '类型的节点特征已存在，跳过生成, 行数为:', tmp_aim_node_UID_pandas.shape[0])
                    print('-----------------------------------------------------------')

                    continue
            ##############################################################################################
            # 上传至pyspark，转为rdd
            tmp_node_table_schema = StructType([StructField(tmp_node_class + '_UID', StringType(), True)])
            tmp_nodes_with_features_rdd = Spark_Session.createDataFrame(tmp_aim_node_UID_pandas, tmp_node_table_schema)

            # 该节点涉及的全部特征
            tmp_all_useful_feature_cols_list = []

            # 基于全部的节点，依次提取全部的特征表中对应的数据，
            for tmp_feature_table_Info_dict in Feature_Dataset_Config_dict[tmp_node_class]['Feature_Data_List']:
                tmp_feature_table_name = tmp_feature_table_Info_dict['Table_Name']
                tmp_aim_column_name = tmp_feature_table_Info_dict['UID']

                tmp_sql_command = """
                            SELECT
                                *
                            FROM
                                """ + tmp_feature_table_name + """
                            WHERE 
                                dt = '""" + tmp_current_table_month_dt + """'
                            """

                tmp_feature_table_rdd = Spark_Session.sql(tmp_sql_command)

                # 将唯一标识列列名改为tmp_node_class + '_UID'
                tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(tmp_aim_column_name, tmp_node_class + '_UID')

                # 只保留其中有效的列（entity_id加数值类型的列，*待优化，数据格式标准化后改为entity_id及其之后的列）
                tmp_useful_feature_cols_list = []
                for col, col_type in tmp_feature_table_rdd.dtypes:
                    if col == tmp_node_class + '_UID':
                        continue

                    if 'entity_id' in col:
                        continue

                    if col_type in ['int', 'integer', 'float', 'bigint','double']:
                        tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(col, col + '__' + tmp_feature_table_name.split('.')[-1])
                        tmp_useful_feature_cols_list.append(col + '__' + tmp_feature_table_name.split('.')[-1])
                    elif col_type != 'string':
                        print('-----------------------------------------------------------')
                        print('WARNING:stange_type:', col, col_type)
                        print('-----------------------------------------------------------')

                tmp_feature_table_rdd = tmp_feature_table_rdd.select([tmp_node_class + '_UID'] + tmp_useful_feature_cols_list)

                tmp_all_useful_feature_cols_list.extend(tmp_useful_feature_cols_list)

                print('特征表'+ tmp_feature_table_name + '添加特征数:', len(tmp_useful_feature_cols_list))

                # join两个表
                tmp_nodes_with_features_rdd = tmp_nodes_with_features_rdd.join(tmp_feature_table_rdd, tmp_node_class + '_UID', 'left')

            print(tmp_node_class + '节点涉及的全部特征数:', len(tmp_all_useful_feature_cols_list))
#             print(tmp_node_class + '节点涉及的全部特征:', tmp_all_useful_feature_cols_list)

            # 默认是0
            tmp_nodes_with_features_rdd = tmp_nodes_with_features_rdd.fillna(0)

            tmp_nodes_with_features_pandas = tmp_nodes_with_features_rdd.toPandas()

            tmp_feature_pd_row_num = tmp_nodes_with_features_pandas.shape[0]

            # 去重
            tmp_nodes_with_features_pandas = tmp_nodes_with_features_pandas.drop_duplicates([tmp_node_class + '_UID'], ignore_index = True)
            if tmp_feature_pd_row_num != tmp_nodes_with_features_pandas.shape[0]:
                print('-----------------------------------------------------------')
                print('WARNING:' + tmp_node_class + '节点涉及的特征表中存在UID重复')
                print('-----------------------------------------------------------')

            # 获取各特征列的最大值、最小值、均值和std
            sub_tmp_pd = tmp_nodes_with_features_pandas.iloc[:, 1:]
            sub_tmp_pd_min = sub_tmp_pd.min()
            sub_tmp_pd_max = sub_tmp_pd.max()
            sub_tmp_pd_mean = sub_tmp_pd.mean()
            sub_tmp_pd_std = sub_tmp_pd.std()

            # 检验特征是否都为同一值
            sub_tmp_pd_all_same_column = sub_tmp_pd_max[sub_tmp_pd_min == sub_tmp_pd_max]
            if sub_tmp_pd_all_same_column.shape[0] > 0:
                print('-----------------------------------------------------------')
                print('WARNING:存在特征列全部行都是一个值，具体情况如下')
                print(sub_tmp_pd_all_same_column)
                print('-----------------------------------------------------------')

            # 进行归一化和标准化
            sub_tmp_pd_norm = ((sub_tmp_pd - sub_tmp_pd_min) / (sub_tmp_pd_max - sub_tmp_pd_min))
            tmp_nodes_with_features_norm_pd = pd.concat([tmp_nodes_with_features_pandas.iloc[:, 0:1], sub_tmp_pd_norm], axis = 1)
            tmp_nodes_with_features_norm_pd = tmp_nodes_with_features_norm_pd.fillna(0) 

            sub_tmp_pd_std = ((sub_tmp_pd - sub_tmp_pd_mean) / sub_tmp_pd_std)
            tmp_nodes_with_features_std_pd = pd.concat([tmp_nodes_with_features_pandas.iloc[:, 0:1], sub_tmp_pd_std], axis = 1)
            tmp_nodes_with_features_std_pd = tmp_nodes_with_features_std_pd.fillna(0)    

            # 再依次保存归一化、标准化和原始情况的结果
            tmp_nodes_with_features_norm_pd.to_pickle(tmp_monthly_output_node_feature_dir + tmp_node_class + '_Norm.pkl')
            tmp_nodes_with_features_std_pd.to_pickle(tmp_monthly_output_node_feature_dir + tmp_node_class + '_Std.pkl')
            tmp_nodes_with_features_pandas.to_pickle(tmp_output_data_node_feature_file)

            print('完成' + tmp_node_class + '类型的节点特征的生成, 行数为:', tmp_nodes_with_features_pandas.shape[0])
        
    return