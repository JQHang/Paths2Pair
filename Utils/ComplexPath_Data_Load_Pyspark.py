# 加入Aim_Metapath_list的选项（如果有值，且不为空就以此为准）
# 分别保存各组groupby后的结果
# 加入只保留部分有效特征的功能（用config）

import numpy as np
import math
import pandas as pd
import os
import gc
import json

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from Utils.utils import mkdir
import Utils.Pyspark_utils
from Utils.Pyspark_utils import Pyspark_String_to_Vector, Pyspark_Merge_Vectors, Pyspark_Fill_Null_Vectors, Data_In_HDFS_Path
from Utils.Pyspark_utils import hdfs_file_exists, hdfs_read_marker_file
from Utils.Pyspark_utils import hdfs_create_marker_file, DEFAULT_STORAGE_LEVEL

from pyspark.sql.types import *
from pyspark.sql.functions import udf, lit
from pyspark.sql.functions import col, row_number
from pyspark.sql.functions import broadcast
from pyspark.ml.linalg import SparseVector, DenseVector, VectorUDT

"""
作用：
    从线上读取指定目标点对应的特征信息和各特征的总结信息，并进行预处理后返回指定结果

输入：
    Feature_Table_Config_list:目标特征表的配置信息
    tmp_Aim_Feature_Table_dt_list：目标时间范围
    
返回值：
    目标节点的特征值
"""
def Single_ComplexPath_Data_Load(Spark_Session, Feature_Table_Config_list, tmp_Aim_Feature_Table_dt_list, Aim_Node_type, Aim_Node_UID_df, 
                       feature_vector_columns = ['Feature_Raw', 'Feature_Normalized']):
    print('Read Features from Complex-Path')
    
    print('Feature_Table_Config_list:', Feature_Table_Config_list)
    
    feature_columns_comment = []
    
    for tmp_config_index in range(len(Feature_Table_Config_list)):
        tmp_feature_table_Info_dict = Feature_Table_Config_list[tmp_config_index]
        
        tmp_feature_table_name = tmp_feature_table_Info_dict['Table_Name']
        tmp_aim_column_name = tmp_feature_table_Info_dict['UID']

        print('Read Complex-Path Agg Table:', tmp_feature_table_name)
        
        tmp_agg_data_path = Data_In_HDFS_Path + "/ComplexPath_Aggregation/" + tmp_feature_table_name
        tmp_dt_paths = [tmp_agg_data_path + f"/dt={dt}" for dt in tmp_Aim_Feature_Table_dt_list]
        
        print('Paths:', tmp_dt_paths)
        
        # Read Feature Comment
        tmp_feature_comment_content = hdfs_read_marker_file(tmp_dt_paths[0], '_Feature_Comment')

        if tmp_feature_comment_content is not None:
            try:
                tmp_feature_comment_list = json.loads(tmp_feature_comment_content)
                feature_columns_comment.extend(tmp_feature_comment_list)
                print('First Five Feature Comments: ', tmp_feature_comment_list[0:5])
            except Exception as e:
                print('Feature Comments not in right style')
        else:
            print('Feature Comments not exist')
            
        tmp_feature_table_rdd = Spark_Session.read.option("basePath", tmp_agg_data_path).parquet(*tmp_dt_paths)
        
        tmp_feature_table_rdd = tmp_feature_table_rdd.select(tmp_aim_column_name, *feature_vector_columns, "dt")
        
        # Delete null column
        tmp_feature_table_rdd = tmp_feature_table_rdd.filter(col(tmp_aim_column_name).isNotNull())
        
        # Uniform the column name of the target node
        tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(tmp_aim_column_name, Aim_Node_type + '_UID')
        
        # Keep useful data
        tmp_sub_feature_table_rdd = tmp_feature_table_rdd.join(Aim_Node_UID_df, 
                                             (tmp_feature_table_rdd[Aim_Node_type + '_UID'] ==  
                                              Aim_Node_UID_df[Aim_Node_type + '_UID']) &
                                             (tmp_feature_table_rdd['dt'] ==  
                                              Aim_Node_UID_df['Feature_Time']), 'inner').\
                                              drop(Aim_Node_UID_df[Aim_Node_type + '_UID'])

        # drop dt(Use Feature_Time Instead)
        tmp_sub_feature_table_rdd = tmp_sub_feature_table_rdd.drop('dt')
        
        tmp_sub_feature_table_rdd = tmp_sub_feature_table_rdd.repartition(Aim_Node_type + '_UID')
        
        tmp_sub_feature_table_rdd = tmp_sub_feature_table_rdd.persist(DEFAULT_STORAGE_LEVEL)
        
        # Get the vector length
        tmp_filtered_df = tmp_sub_feature_table_rdd.select(feature_vector_columns[0]).filter(col(feature_vector_columns[0]).isNotNull())
        tmp_limited_df = tmp_filtered_df.limit(1)
        tmp_results = tmp_limited_df.collect()
        if tmp_results:
            tmp_vector_length = len(tmp_results[0][0]) if tmp_results[0][0] else 0
        else:
            tmp_vector_length = None
        print('The length of the feature vector is:', tmp_vector_length)
    
        # Combine with previous data
        if tmp_config_index == 0:
            tmp_aim_node_agg_feature_rdd = tmp_sub_feature_table_rdd
            
            # Fill null row
            tmp_aim_node_agg_feature_rdd = Pyspark_Fill_Null_Vectors(Spark_Session, tmp_aim_node_agg_feature_rdd, feature_vector_columns,
                                                   tmp_vector_length)
        else:
            # Add suffix to the feature vector column
            for tmp_col_name in feature_vector_columns:
                tmp_sub_feature_table_rdd = tmp_sub_feature_table_rdd.withColumnRenamed(tmp_col_name, tmp_col_name + '_Add')
            
            tmp_aim_node_agg_feature_rdd = tmp_aim_node_agg_feature_rdd.join(tmp_sub_feature_table_rdd, [Aim_Node_type + '_UID', 
                                                                          'Feature_Time'], 'outer')
            
            # Fill null row
            tmp_aim_node_agg_feature_rdd = Pyspark_Fill_Null_Vectors(Spark_Session, tmp_aim_node_agg_feature_rdd, 
                                                   [col + '_Add' for col in feature_vector_columns],
                                                   tmp_vector_length)
            
            for tmp_col_name in feature_vector_columns:
                # Merge the vector columns
                tmp_aim_node_agg_feature_rdd = Pyspark_Merge_Vectors(Spark_Session, tmp_aim_node_agg_feature_rdd, tmp_col_name, 
                                                    tmp_col_name + '_Add')
    
    print('Finish reading feature')
    
    return tmp_aim_node_agg_feature_rdd, feature_columns_comment

"""
作用：
    从线上读取聚合后的pandas数据（包含标签列 + 处理后的元路径特征列）。

输入：

    
返回值：
    输出结果直接存储在对应文件夹中
"""
def All_ComplexPath_Data_Load(Spark_Session, Aim_UID_info_dict, Feature_Data_From_Online_Config_dict, Data_Store_dir,
                    regenerate = False, ComplexPath_drop_list = [], ComplexPath_Column_drop_dict = {}, 
                    feature_vector_columns = ['Feature_Raw', 'Feature_Normalized']):
    
    tmp_aggregate_data_start_time = datetime.now()
    
    Complex_Path_Pyspark_Data_Dict = {}
    
    ##################################################################################################################
    # Read target node info
    tmp_aim_entity_rdd = Aim_UID_info_dict['Data']
    Aim_Node_type = Aim_UID_info_dict['Node_Type']
    Aim_Node_UID_Name = Aim_Node_type + '_UID'

    # 只保留rdd中的目标节点列，去缩减特征表
    tmp_aim_node_UID_rdd = tmp_aim_entity_rdd.select([Aim_Node_UID_Name, 'Feature_Time'])
    
    # Drop duplicates
    tmp_aim_node_UID_rdd = tmp_aim_node_UID_rdd.dropDuplicates([Aim_Node_UID_Name, 'Feature_Time'])
    tmp_aim_node_UID_rdd = tmp_aim_node_UID_rdd.persist(DEFAULT_STORAGE_LEVEL)
    
    # Get all the complex-paths
    tmp_complex_path_name_list = list(Feature_Data_From_Online_Config_dict['Meta_Path_Feature_Table_List'].keys())
    tmp_complex_path_name_list.sort()

    print('ComplexPaths to process:', tmp_complex_path_name_list)
    
    tmp_Aim_Feature_Table_dt_list = Aim_UID_info_dict['Data_dt_list']
    
    print('All the feature time:', tmp_Aim_Feature_Table_dt_list)
    ###########################################################################################################################
    print('----------------------------------------------------------------------------')
    print('Extract features for the target nodes')
    
    tmp_output_pyspark_df_store_dir = Data_Store_dir + '/Target_Node'
    print('Output dir in HDFS:', tmp_output_pyspark_df_store_dir)
    
    if regenerate or not hdfs_file_exists(tmp_output_pyspark_df_store_dir + '/_SUCCESS'):
        tmp_start_node_start_time = datetime.now()
        
        tmp_Feature_Table_Config_list = Feature_Data_From_Online_Config_dict['Start_Node_Feature_Table_List']
        tmp_aim_node_agg_feature_rdd, feature_columns_comment = Single_ComplexPath_Data_Load(Spark_Session, tmp_Feature_Table_Config_list, 
                                                                 tmp_Aim_Feature_Table_dt_list, Aim_Node_type, 
                                                                 tmp_aim_node_UID_rdd, feature_vector_columns)
        
        tmp_aim_node_agg_feature_rdd.persist(DEFAULT_STORAGE_LEVEL)
        
        # Store to target dir
        tmp_aim_node_agg_feature_rdd.write.mode("overwrite").parquet(tmp_output_pyspark_df_store_dir)
        
        # Store the feature column comment
        if len(feature_columns_comment) != 0:
            tmp_feature_comment_str = json.dumps(feature_columns_comment)
            hdfs_create_marker_file(tmp_output_pyspark_df_store_dir, '_Feature_Comment', tmp_feature_comment_str)
        
        tmp_start_node_end_time = datetime.now()
    
        print('Time cost:', (tmp_start_node_end_time - tmp_start_node_start_time))
        
    else:
        tmp_aim_node_agg_feature_rdd = Spark_Session.read.parquet(tmp_output_pyspark_df_store_dir)
        
        # Read Feature Comment
        tmp_feature_comment_str = hdfs_read_marker_file(tmp_output_pyspark_df_store_dir, '_Feature_Comment')
        if tmp_feature_comment_str is not None:
            try:
                feature_columns_comment = json.loads(tmp_feature_comment_str)
            except Exception as e:
                print('Feature Comments not in right style')
        else:
            feature_columns_comment = []
        print('File Existed')
        
    Complex_Path_Pyspark_Data_Dict['Target_Node'] = tmp_aim_node_agg_feature_rdd
    Complex_Path_Pyspark_Data_Dict['Target_Node_Comment'] = feature_columns_comment
    ###########################################################################################################################
    print('----------------------------------------------------------------------------')
    print('Process the following complex-paths:', tmp_complex_path_name_list)

    # 依次读取各元路径对应的数据
    for tmp_complex_path_name in tmp_complex_path_name_list:
        tmp_meta_path_start_time = datetime.now()
        
        # 跳过不需要的元路径
        if tmp_complex_path_name in ComplexPath_drop_list:
            print('Skip Complex-path:', tmp_complex_path_name)
            continue

        print('Process Complex-Path:', tmp_complex_path_name)

        Complex_Path_Pyspark_Data_Dict[tmp_complex_path_name] = []
        Complex_Path_Pyspark_Data_Dict[tmp_complex_path_name + '_Comment'] = []
        
        # 获得对应元路径的配置信息
        tmp_meta_path_info = Feature_Data_From_Online_Config_dict['Meta_Path_Feature_Table_List'][tmp_complex_path_name]
        
        # 依次处理元路径中各列的信息
        for tmp_column_i in range(len(tmp_meta_path_info)):
            if tmp_complex_path_name in ComplexPath_Column_drop_dict and tmp_column_i in ComplexPath_Column_drop_dict[tmp_complex_path_name]:
                print('Skip the ', str(tmp_column_i) + '-th column of the complex-path', tmp_complex_path_name)
                continue
            
            tmp_output_pyspark_df_store_dir = Data_Store_dir + '/' + tmp_complex_path_name + '/' + str(tmp_column_i)
            print('Output dir in HDFS:', tmp_output_pyspark_df_store_dir)
            
            if regenerate or not hdfs_file_exists(tmp_output_pyspark_df_store_dir + '/_SUCCESS'):
                tmp_meta_path_column_start_time = datetime.now()

                # 获取对应列节点类型
                tmp_column_node_class = tmp_meta_path_info[tmp_column_i]["Node_class"]

                # 将特征表信息转化为指定格式
                tmp_Feature_Table_Config_list = []
                for tmp_aim_feature_table_i in range(len(tmp_meta_path_info[tmp_column_i]["Feature_Table_List"])):
                    tmp_feature_table_name = tmp_meta_path_info[tmp_column_i]["Feature_Table_List"][tmp_aim_feature_table_i]
                    tmp_aim_column_name = 'Start_Column'

                    tmp_Feature_Table_Config_list.append({"Table_Name":tmp_feature_table_name, 
                                              "UID": tmp_aim_column_name})

                # Get the target data
                tmp_aim_node_agg_feature_rdd, feature_columns_comment = Single_ComplexPath_Data_Load(Spark_Session, tmp_Feature_Table_Config_list, 
                                                        tmp_Aim_Feature_Table_dt_list, Aim_Node_type, 
                                                        tmp_aim_node_UID_rdd, feature_vector_columns)
                
                tmp_aim_node_agg_feature_rdd.persist(DEFAULT_STORAGE_LEVEL)
                
                # Store Dir
                tmp_aim_node_agg_feature_rdd.write.mode("overwrite").parquet(tmp_output_pyspark_df_store_dir)
                
                # Store the feature column comment
                if len(feature_columns_comment) != 0:
                    tmp_feature_comment_str = json.dumps(feature_columns_comment)
                    hdfs_create_marker_file(tmp_output_pyspark_df_store_dir, '_Feature_Comment', tmp_feature_comment_str)
            
                tmp_meta_path_column_end_time = datetime.now()

                print('完成对元路径', tmp_complex_path_name, '的第', tmp_column_i, '列的结果的生成，共花费时间:', 
                     (tmp_meta_path_column_end_time - tmp_meta_path_column_start_time))
                print('----------------------------------------------------------------------------')
            else:
                tmp_aim_node_agg_feature_rdd = Spark_Session.read.parquet(tmp_output_pyspark_df_store_dir)
            
                # Read Feature Comment
                tmp_feature_comment_str = hdfs_read_marker_file(tmp_output_pyspark_df_store_dir, '_Feature_Comment')
                if tmp_feature_comment_str is not None:
                    try:
                        feature_columns_comment = json.loads(tmp_feature_comment_str)
                    except Exception as e:
                        print('Feature Comments not in right style')
                else:
                    feature_columns_comment = []
                    
                tmp_aim_node_agg_feature_rdd.persist(DEFAULT_STORAGE_LEVEL)
                
                print('File Existed')
            
            Complex_Path_Pyspark_Data_Dict[tmp_complex_path_name].append(tmp_aim_node_agg_feature_rdd)
            Complex_Path_Pyspark_Data_Dict[tmp_complex_path_name + '_Comment'].append(feature_columns_comment)
            
        tmp_meta_path_end_time = datetime.now()
        print('完成对元路径', tmp_complex_path_name, '的全部列的结果的生成，共花费时间:', (tmp_meta_path_end_time - tmp_meta_path_start_time))
        print('----------------------------------------------------------------------------')
    
    tmp_aggregate_data_end_time = datetime.now()
    
    print('完成全部数据生成，总共花费时间:', (tmp_aggregate_data_end_time - tmp_aggregate_data_start_time))
    print('----------------------------------------------------------------------------')
    
    return Complex_Path_Pyspark_Data_Dict
