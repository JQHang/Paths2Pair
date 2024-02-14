import Utils.Pyspark_utils
from Utils.Pyspark_utils import Data_In_HDFS_Path, estimate_partitions
from Utils.Pyspark_utils import sample_rdd_from_aim_row, sample_rdd_from_aim_range
from Utils.Pyspark_utils import Groupby_Feature_Table, sanitize_column_names
from Utils.Pyspark_utils import Pyspark_Create_Table, Upload_RDD_Data_to_Database, Preprocess_Numerical_Data
from Utils.Pyspark_utils import hdfs_create_marker_file, DEFAULT_STORAGE_LEVEL
from Utils.Pyspark_utils import check_numeric_columns, hdfs_read_marker_file, hdfs_file_exists
from Utils.Pyspark_utils import Spark_Random_N_Sample, Spark_Top_N_Sample, Spark_Threshold_N_Sample
from Utils.Pyspark_utils import pyspark_feature_aggregation

from Utils.Decorator import Time_Costing

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import SparseVector, DenseVector, VectorUDT, Vectors
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import IntegerType, FloatType, DoubleType, DecimalType, LongType, ShortType, ByteType
from pyspark.sql.types import *
from pyspark.sql.functions import broadcast, col
from pyspark.storagelevel import StorageLevel

import pandas as pd
import numpy as np
import math
import os
import json
from tqdm import tqdm
from datetime import datetime
from functools import reduce

def read_node_table(spark, node_std_name, node_table_config, node_times, target_node = None, feat_cols_output_type = 'columns', 
              data_check = False):
    """
    读取节点表中的目标信息
    
    输入：
        节点表配置信息(来源表名，目标节点列名，目标节点标准化后的列名，目标节点类型，目标节点特征列list-可选)
        所需全部时间点
    
    输出：
       node_table['data']: 返回的节点表数据
       node_table['node_column']: 节点列
       node_table['feature_columns']: 节点特征列
       node_table['feature_comments']: 节点注释
    """
    print('Read node table:', node_table_config['node_table_name'], 'at times:', node_times)

    if "node_table_source" in node_table_config and node_table_config["node_table_source"] == "HDFS":
        table_data_path = Data_In_HDFS_Path + "/ComplexPath_Aggregation/" + node_table_config['node_table_name']

        table_data_paths = [table_data_path + f"/dt={dt}" for dt in node_times]

        print('Read table from paths:', table_data_paths)

        node_table_df = spark.read.option("basePath", table_data_path).parquet(*table_data_paths)

    else:
        node_table_df = spark.table(node_table_config['node_table_name'])

        node_table_df = node_table_df.filter(node_table_df.dt.isin(node_times))
        
    # 将目标节点原始列名转为标准名称
    node_table_df = node_table_df.withColumnRenamed(node_table_config['node_raw_name'], node_std_name)
    
    # 将时间列改为标准列名
    node_table_df = node_table_df.withColumnRenamed('dt', 'Feature_Time')
    
    # 获得目标特征列
    if 'target_feature_columns' in node_table_config:
        feature_columns = node_table_config['target_feature_columns']
    else:
        feature_columns, feature_columns_comments = check_numeric_columns(node_table_df.schema)
    
    print("The number of feature columns in this node table:", len(feature_columns))
    
    # 只保留目标节点列和特征列
    node_table_df = node_table_df.select([node_std_name, 'Feature_Time'] + feature_columns)
    
    # 基于目标节点列去重
    node_table_df = node_table_df.dropDuplicates([node_std_name, "Feature_Time"])
    
    node_table_df.persist(DEFAULT_STORAGE_LEVEL)
    
    ##############################################################################################
    # Check whether the feature data isValid
    if data_check:
        if node_table_df.limit(1).count() == 0:
            # Check isEmpty
            print('Error: Node Table', node_table_config['node_table_name'], 'is empty')
        else:
            # Get max and min for each feature column
            feat_summary_df = node_table_df.select(feature_columns).summary("min", "max")

            # Transfer the result to dict
            min_values = feat_summary_df.filter(col("summary") == "min").drop("summary").collect()[0].asDict()
            max_values = feat_summary_df.filter(col("summary") == "max").drop("summary").collect()[0].asDict()

            # check whether they are equal and keep the result
            invalid_cols_values = {col_name: min_values[col_name] for col_name in min_values.keys() if min_values[col_name] == max_values[col_name]}

            if len(invalid_cols_values.keys()) > 0:
                print('ERROR: Node Table ', node_table_config['node_table_name'], 'has invalid columns at times', node_times, 
                'The numbers in the column are all the same, and the column name and value are:')
                print(invalid_cols_values)

    ##############################################################################################
    # 保存输出结果
    node_table = {}
    node_table['data'] = node_table_df
    node_table['node_column'] = node_std_name
    node_table['feature_columns'] = feature_columns
    node_table['feature_comments'] = [x + f"({node_table_config['node_table_name']})"for x in feature_columns_comments]
    
    ##############################################################################################
    # 如果有目标节点，则只为目标节点对应的节点列，添加上对应数据
    if target_node is not None:
        # 保证节点表中的节点列列名和目标节点表中的一致
        node_table['data'] = node_table['data'].withColumnRenamed(node_std_name, target_node['node_column'])
        
        # 单独的节点表join时，只输出节点列及对应特征，防止合并多个节点表结果时的问题
        # （理论上不该有单独调用读取单节点表的函数的情况，应该统一调用读取多节点表的函数）
        target_node_table_df = target_node['data'].select([target_node['node_column'], "Feature_Time"])
        
        # 为节点表中的对应节点列添加数据
        target_node_table_df = target_node_table_df.join(node_table['data'], [target_node['node_column'], "Feature_Time"], 'left')
        
        target_node_table_df.persist(DEFAULT_STORAGE_LEVEL)
        
        target_node_table_count = target_node_table_df.count()
        print('The number of target nodes in this node table:', target_node_table_count)
        
        # 基于结果重新分区
        best_partitions = estimate_partitions(target_node_table_df, approximate_row_count = target_node_table_count)
        print("best partition numbers:", best_partitions)
        target_node_table_df = target_node_table_df.repartition(best_partitions, target_node['node_column'], "Feature_Time")
    
        node_table_df.unpersist()
        
        node_table['data'] = target_node_table_df
    
    # 将空特征作为0
    node_table['data'] = node_table['data'].fillna(0)
    
    ##############################################################################################
    # 如果输出类型为vector，则将特征列转化为一个vector列，并更新结果
    if feat_cols_output_type == 'vector':
        feature_vector_assembler = VectorAssembler(inputCols = node_table['feature_columns'], outputCol = f'Feature_Vector')
        vector_node_table_df = feature_vector_assembler.transform(node_table['data'])
        vector_node_table_df = vector_node_table_df.drop(*node_table['feature_columns'])

        vector_node_table_df.persist(DEFAULT_STORAGE_LEVEL)
        
        print('The number of target nodes in this vector node table:', vector_node_table_df.count())
        
        node_table['data'].unpersist()
        
        node_table['data'] = vector_node_table_df
        node_table['feature_columns'] = ['Feature_Vector']
        
    return node_table

def read_list_node_tables(spark, node_std_name, node_table_configs, node_times, target_node = None, feat_cols_output_type = 'columns', 
                  result_dir = None):
    """
    合并多个节点表中包含的全部特征，并以指定形式输出
    
    输入：
       node_std_name: 节点列的统一命名
       node_table_configs: 一串节点特征表的list
       node_times: 节点表对应的时间
       feat_cols_output_type: 输出列的类型
       target_node: 目标节点
       
    输出：
       node_table['data']: 返回的节点表数据
       node_table['node_column']: 节点列
       node_table['feature_columns']: 节点特征列
       node_table['feature_comments']: 节点注释
       
    """  
    # 如果result_dir不为空，且下面的node_std_name下的文件夹有各个node_times的结果，则直接读取并返回
    
    # 给target_node去重
    distinct_target_node = {}
    distinct_target_node['node_column'] = target_node['node_column']
    distinct_target_node['data'] = target_node['data'].select([target_node['node_column'], "Feature_Time"]).distinct()
    distinct_target_node['data'].persist(DEFAULT_STORAGE_LEVEL)
    
    print('去重后的目标点数目:', distinct_target_node['data'].count())
    
    # 依次读取各个特征表里的数据
    node_tables = []
    for node_table_index, node_table_config in enumerate(node_table_configs):
        
        # 读取对应数据
        node_table = read_node_table(spark, node_std_name, node_table_config, node_times, distinct_target_node, feat_cols_output_type)
        
        # 给特征列加入前缀，防止重复，并更新对应的feature_columns
        feature_name_prefix = f'Node_Table_{node_table_index}_'
        new_columns = []
        for column in node_table['data'].columns:
            if column in node_table['feature_columns']:
                new_column = col(column).alias(feature_name_prefix + column)
            else:
                new_column = col(column)
            new_columns.append(new_column)
        node_table['data'] = node_table['data'].select(new_columns)
        node_table['feature_columns'] = [feature_name_prefix + x for x in node_table['feature_columns']]
            
        # 记录结果
        node_tables.append(node_table)
    
    print(f"Combining result from {len(node_table_configs)} node tables")
    
    # 合并各个节点表的结果
    combined_node_table = {}
    combined_node_table['feature_columns'] = []
    combined_node_table['feature_comments'] = []
    if target_node is not None:
        # 如果有target_node则以此为基础进行left_join来合并各个节点表
        combined_node_table_df = target_node['data'].select("*")
        
        for index, node_table in enumerate(node_tables):
            node_table_df = node_table['data'].withColumnRenamed(node_std_name, target_node['node_column'])
            
            combined_node_table_df = combined_node_table_df.join(node_table_df, on = [target_node['node_column'], 'Feature_Time'], 
                                                how = 'left')
            
            # 记录添加的特征列的信息
            combined_node_table['feature_columns'].extend(node_table['feature_columns'])
            combined_node_table['feature_comments'].extend(node_table['feature_comments'])
    
        combined_node_table['data'] = combined_node_table_df
        combined_node_table['node_column'] = target_node['node_column']
        
    else:
        # 若没有，则进行outer_join来合并各个节点表
        combined_node_table_df = node_tables[0]['data'].select("*")
        for index, node_table in enumerate(node_tables[1:]):
            combined_node_table_df = combined_node_table_df.join(node_table['data'], on = [node_std_name, 'Feature_Time'], how = 'outer')
            
            # 记录添加的特征列的信息
            combined_node_table['feature_columns'].extend(node_table['feature_columns'])
            combined_node_table['feature_comments'].extend(node_table['feature_comments'])
            
        combined_node_table['node_column'] = node_std_name
        combined_node_table['data'] = combined_node_table_df
    
    # 将空特征作为0
    combined_node_table['data'] = combined_node_table['data'].fillna(0)
    
    # 如果输出的是vector类型
    if feat_cols_output_type == 'vector':
        # 依次给对应vector列补全相同长度的全0的vector
        for index, node_table in enumerate(node_tables):
            # 获得对应特征vector列列名
            feature_vector_column = node_tables[index]['feature_columns'][0]

            # 获得对应的vector长度
            vector_length = len(node_tables[index]['feature_comments'])
            
            # 创建指定长度的全为0的vector
            default_vector = SparseVector(vector_length, {})
            
            # 给对应列补全相同长度的全0的vector
            fill_udf = udf(lambda v: default_vector, VectorUDT())
            combined_node_table['data'] = combined_node_table['data'].withColumn(feature_vector_column, fill_udf(col(feature_vector_column)))

        # 合并全部vector
        feature_vector_assembler = VectorAssembler(inputCols = combined_node_table['feature_columns'], outputCol = f'Feature_Vector')
        combined_node_table['data'] = feature_vector_assembler.transform(combined_node_table['data'])
        combined_node_table['data'] = combined_node_table['data'].drop(*combined_node_table['feature_columns'])
        
        combined_node_table['feature_columns'] = ['Feature_Vector']
    
    combined_node_table['data'].persist(DEFAULT_STORAGE_LEVEL)
        
    print('The number of the nodes in these node tables:', combined_node_table['data'].count())
    
    # 优化分区数
    
    return combined_node_table

def read_edge_table(spark, edge_table_config, edge_times, target_node = None, feat_cols_output_type = 'columns'):
    """
    读取关系表，并基于node_limit和edge_limit过滤出有效边
    """
    # edge_limit里面有属性限制和关联度-Connectivity
    print('Read edge table:', edge_table_config['edge_table_name'], 'at times:', edge_times)
    
    if "edge_table_source" in edge_table_config and edge_table_config["edge_table_source"] == "HDFS":
        table_data_path = Data_In_HDFS_Path + "/ComplexPath_Aggregation/" + edge_table_config['edge_table_name']
        table_data_paths = [table_data_path + f"/dt={dt}" for dt in edge_times]
        
        edge_table_df = spark.read.option("basePath", table_data_path).parquet(*table_data_paths)

    else:
        edge_table_df = spark.table(edge_table_config['edge_table_name'])
        
        edge_table_df = edge_table_df.filter(edge_table_df.dt.isin(edge_times))
    
    # ID columns
    edge_id_columns = []
    
    # Change the node column and time column name
    head_node_std_name = edge_table_config['head_node_std_name']
    edge_table_df = edge_table_df.withColumnRenamed(edge_table_config['head_node_raw_name'], head_node_std_name)
    edge_table_df = edge_table_df.filter(col(head_node_std_name).isNotNull() & (col(head_node_std_name) != ""))
    edge_id_columns.append(head_node_std_name)
    
    if 'tail_node_std_name' in edge_table_config:
        tail_node_std_name = edge_table_config['tail_node_std_name']
        edge_table_df = edge_table_df.withColumnRenamed(edge_table_config['tail_node_raw_name'], tail_node_std_name)
        edge_table_df = edge_table_df.filter(col(tail_node_std_name).isNotNull() & (col(tail_node_std_name) != ""))
        edge_id_columns.append(tail_node_std_name)
    else:
        for tail_node_index in range(len(edge_table_config['tail_node_raw_names'])):
            tail_node_raw_name = edge_table_config['tail_node_raw_names'][tail_node_index]
            tail_node_std_name = edge_table_config['tail_node_std_names'][tail_node_index]
            
            edge_table_df = edge_table_df.withColumnRenamed(tail_node_raw_name, tail_node_std_name)
            edge_table_df = edge_table_df.filter(col(tail_node_std_name).isNotNull() & (col(tail_node_std_name) != ""))
            edge_id_columns.append(tail_node_std_name)
        
    edge_table_df = edge_table_df.withColumnRenamed('dt', 'Feature_Time')
    edge_id_columns.append('Feature_Time')
    
    # Change column name with specific characters
    edge_table_df = sanitize_column_names(edge_table_df)
    
    # Get feature information for the edge
    if 'edge_target_feature_columns' in edge_table_config and isinstance(edge_table_config['edge_target_feature_columns'], list):
        feature_columns = edge_table_config['edge_target_feature_columns']
    else:
        feature_columns, feature_columns_comments = check_numeric_columns(edge_table_df.schema)
    
    # Only Keep the target node columns, time column and edge feature columns
    edge_table_df = edge_table_df.select(edge_id_columns + feature_columns)
    
    ##############################################################################################
    # Only keep edges start from the start node, if provided
    if target_node is not None:
        target_node_df = target_node['data']
        target_node_column_name = target_node['node_column']
        
        target_node_df = target_node_df.withColumnRenamed(target_node_column_name, head_node_std_name)
        target_node_df = target_node_df.select([head_node_std_name, "Feature_Time"]).distinct()
        
        edge_table_df = edge_table_df.join(target_node_df, [head_node_std_name, "Feature_Time"], 'inner')
        
        edge_table_df = edge_table_df.repartition(head_node_std_name, "Feature_Time")
        
    # Groupby edges with the same node and accumulate the edge features
    if len(feature_columns) > 0:
        edge_table_df = Groupby_Feature_Table(spark, edge_table_df, edge_id_columns, feature_columns, ['SUM'])
        for feat_col in feature_columns:
            # 待优化命名**
            edge_table_df = edge_table_df.withColumnRenamed('SUM_' + feat_col, feat_col)
    else:
        edge_table_df = edge_table_df.distinct()
        
    ##############################################################################################
    # Add feature limitation
    if 'Edge_Feature_Limits' in edge_table_config and edge_table_config['Edge_Feature_Limits'] != '':
        edge_table_df = edge_table_df.where(edge_table_config['Edge_Feature_Limits'])
        print('Edge Limitation:', edge_table_config['Edge_Feature_Limits'])
    
    if 'Node_Feature_Limits' in edge_table_config and len(edge_table_config['Node_Feature_Limits']) > 0:
        # 设定Node Feature Limits的目标节点
        target_node_limit = {}
        target_node_limit['data'] = edge_table_df
        target_node_limit['node_column'] = edge_table_config['Node_Feature_Limits']['node_std_name']
        
        # 获取对应的节点类型
        node_feature_table = read_node_table(spark, target_node_limit['node_column'], edge_table_config['Node_Feature_Limits']['node_tables'], 
                                 node_times, target_node_limit, 'columns')
        
        print('Node Limits:', edge_table_config['Node_Feature_Limits']['limit'])
        edge_table_df = node_feature_table['data'].where(edge_table_config['Node_Feature_Limits']['limit'])

        # Only keep the raw columns in the edge table
        edge_table_df = edge_table_df.select(edge_id_columns + feature_columns)
    
    ##############################################################################################
    # Add edge neighbor limits
    if 'Edge_Neighbor_Limits' in edge_table_config:
        edge_neighbor_limit_type = 'Random_N'
        if 'Type' in edge_table_config['Edge_Neighbor_Limits']:
            edge_neighbor_limit_type = edge_table_config['Edge_Neighbor_Limits']['Type']
            
        edge_neighbor_limit_max_num = edge_table_config['Edge_Neighbor_Limits']['Max_Num']
        
        edge_neighbor_limit_feat_columns = []
        if 'Feat_Columns' in edge_table_config['Edge_Neighbor_Limits']:
            edge_neighbor_limit_feat_columns = edge_table_config['Edge_Neighbor_Limits']['Feat_Columns']
        
        print(f'{edge_neighbor_limit_type} Sampling')
        print(f'Max neighbor count: {edge_neighbor_limit_max_num}, Target feature columns: {edge_neighbor_limit_feat_columns}')

        if edge_neighbor_limit_type == 'Threshold_N':
            edge_table_df = Spark_Threshold_N_Sample(spark, edge_table_df, [head_node_std_name], edge_neighbor_limit_max_num)
        elif edge_neighbor_limit_type == 'Top_N':
            edge_table_df = Spark_Top_N_Sample(spark, edge_table_df, [head_node_std_name], edge_neighbor_limit_max_num, 
                                    tmp_Max_Sample_Feat_Columns)
        else:
            edge_table_df = Spark_Random_N_Sample(spark, edge_table_df, [head_node_std_name], edge_neighbor_limit_max_num)

    edge_table = {}
    edge_table['data'] = edge_table_df
    edge_table['feature_columns'] = feature_columns
    edge_table['feature_comments'] = feature_columns_comments
    
    return edge_table

# def read_list_edge_tables(spark, edge_table_configs, edge_times, target_node = None, feat_cols_output_type = 'columns'):
#     """
#     读取一组有同样起始和终止点的关系表，合并后再返回
#     """
    
#     return

@Time_Costing
def complex_path_sampling(spark, complex_path_name, complex_path_config, edge_times, result_store_dir, target_node = None, 
                  feat_cols_output_type = 'columns'):
    """
    采样指定的complex path路径上的全部点

    输入：

    返回值：
        
    """
    print("Sampling neighbor nodes for Complex-Path:", complex_path_name)
    
    complex_path_sample_result_base_dir = result_store_dir + f'/Complex_Path_Sampling/{complex_path_name}'
    print("The reulst will be output to:", complex_path_sample_result_base_dir)
    
    unfinished_edge_times = []
    for edge_time in edge_times:
        if not hdfs_file_exists(complex_path_sample_result_base_dir + f"/dt={edge_time}/_SUCCESS"): 
            unfinished_edge_times.append(edge_time)
    
    print("Unfinished times for this complex path:", unfinished_edge_times)
    if len(unfinished_edge_times) == 0:
        print("The complex path sampling result already exists")
        
        complex_path = {}
        
        complex_path['data'] = spark.read.parquet(complex_path_sample_result_base_dir)
        complex_path['data'] = complex_path['data'].withColumnRenamed("dt", "Feature_Time")
        complex_path['data'] = complex_path['data'].persist(DEFAULT_STORAGE_LEVEL)
        
        tmp_str = hdfs_read_marker_file(complex_path_sample_result_base_dir, '_PathConfig')
        complex_path['path_config'] = json.loads(tmp_str)
        
        tmp_str = hdfs_read_marker_file(complex_path_sample_result_base_dir, '_EdgeFeatCols')
        complex_path['edge_feature_columns'] = json.loads(tmp_str)
        
        tmp_str = hdfs_read_marker_file(complex_path_sample_result_base_dir, '_EdgeFeatComments')
        complex_path['edge_feature_comments'] = json.loads(tmp_str)
        
        return complex_path
    
    ###################################################################################################################
    # 补全未写的配置为默认值
    for hop_k, edge_table_config in enumerate(complex_path_config):
        if 'head_node_std_name' not in edge_table_config:
            edge_table_config['head_node_std_name'] = f'Node_{hop_k}'
        
        if 'tail_node_raw_name' in edge_table_config and 'tail_node_std_name' not in edge_table_config:
            edge_table_config['tail_node_std_name'] = f'Node_{hop_k + 1}'
            
        elif 'tail_node_raw_names' in edge_table_config and 'tail_node_std_names' not in edge_table_config:
            edge_table_config['tail_node_std_names'] = []
            for tail_node_index in range(len(tmp_edge_info['tail_node_raw_names'])):
                edge_table_config['tail_node_std_names'].append(f'Node_{hop_k + 1}_{tail_node_index}')
    
    ###################################################################################################################
    edge_tables_feat_info = {}
    edge_target_node = target_node
    for hop_k, edge_table_config in enumerate(complex_path_config):
        # Get the required edge table
        edge_table = read_edge_table(spark, edge_table_config, edge_times, edge_target_node)
        
        # Add preffix to the feature columns
        for edge_feat_index, edge_feat_raw_name in enumerate(edge_table['feature_columns']):
            edge_feat_new_name = f'Hop_{hop_k}_Edge_Table_' + edge_feat_raw_name
            
            edge_table['data'] = edge_table['data'].withColumnRenamed(edge_feat_raw_name, edge_feat_new_name)
            edge_table['feature_columns'][edge_feat_index] = edge_feat_new_name
            
        # Join with previous edge tables
        if hop_k == 0:
            complex_path_df = edge_table['data']
            edge_tables_feat_info['edge_feature_columns'] = [edge_table['feature_columns']]
            edge_tables_feat_info['edge_feature_comments'] = [edge_table['feature_comments']]
        else:
            complex_path_df = old_complex_path_df.join(edge_table['data'], [edge_table_config['head_node_std_name'], "Feature_Time"], "inner")
            edge_tables_feat_info['edge_feature_columns'].append(edge_table['feature_columns'])
            edge_tables_feat_info['edge_feature_comments'].append(edge_table['feature_comments'])
            
        # Repartition the path data based on the tail node
        if hop_k != (len(complex_path_config) - 1):
            complex_path_df = complex_path_df.repartition(complex_path_config[0]['head_node_std_name'], 
                                           complex_path_config[hop_k + 1]['head_node_std_name'], "Feature_Time")
        
            # Renew target node if we have initial target node
            if edge_target_node is not None:
                # persist data
                complex_path_df.persist(DEFAULT_STORAGE_LEVEL)

                # renew the edge start node for the next edge
                edge_target_node = {}
                edge_target_node_column = complex_path_config[hop_k + 1]['head_node_std_name']
                edge_target_node['data'] = complex_path_df.select([edge_target_node_column, "Feature_Time"]).distinct()
                edge_target_node['data'] = edge_target_node['data'].repartition(edge_target_node_column, "Feature_Time")
                edge_target_node['node_column'] = edge_target_node_column
                
                print('Target node count for the next hop:', edge_target_node['data'].count())
                
                if hop_k != 0:
                    old_complex_path_df.unpersist()
                
        # Path Limit
        if 'Path_Feature_Limits' in edge_table_config and edge_table_config['Path_Feature_Limits'] != '':
            complex_path_df = complex_path_df.where(edge_table_config['Path_Feature_Limits'])
            print('Path Limit', edge_table_config['Path_Feature_Limits'])
        
        if 'Path_Neighbor_Limits' in edge_table_config:
            path_start_node_name = complex_path_config[0]['head_node_std_name']
            
            path_neighbor_limit_type = 'Random_N'
            if 'Type' in edge_table_config['Path_Neighbor_Limits']:
                path_neighbor_limit_type = edge_table_config['Path_Neighbor_Limits']['Type']

            path_neighbor_limit_max_num = edge_table_config['Path_Neighbor_Limits']['Max_Num']

            path_neighbor_limit_feat_columns = []
            if 'Feat_Columns' in edge_table_config['Path_Neighbor_Limits']:
                path_neighbor_limit_feat_columns = edge_table_config['Path_Neighbor_Limits']['Feat_Columns']

            print(f'{path_neighbor_limit_type} Sampling') 
            print(f'Max neighbor count: {path_neighbor_limit_max_num}, Target feature columns: {path_neighbor_limit_feat_columns}')

            if path_neighbor_limit_type == 'Threshold_N':
                complex_path_df = Spark_Threshold_N_Sample(spark, complex_path_df, path_start_node_name, path_neighbor_limit_max_num)
            elif path_neighbor_limit_type == 'Top_N':
                complex_path_df = Spark_Top_N_Sample(spark, complex_path_df, path_start_node_name, path_neighbor_limit_max_num, tmp_Max_Sample_Feat_Columns)
            else:
                complex_path_df = Spark_Random_N_Sample(spark, complex_path_df, path_start_node_name, path_neighbor_limit_max_num)
        
        # record the result
        old_complex_path_df = complex_path_df
        
    ###################################################################################################################
    complex_path_df.persist(DEFAULT_STORAGE_LEVEL)
    
    # 显示结果
    complex_path_count = complex_path_df.count()
    print('Complex Path:', complex_path_name, 'Output rows:', complex_path_count, 'Output columns length:', len(complex_path_df.columns))
    
    # 计算最优分区数
    best_partitions = estimate_partitions(complex_path_df, approximate_row_count = complex_path_count)
    print("最优分区数:", best_partitions)
    complex_path_df = complex_path_df.repartition(best_partitions, "Feature_Time", complex_path_config[0]['head_node_std_name'], 
                                   complex_path_config[-1]['head_node_std_name'])
    
    # 保存结果
    complex_path_df.withColumnRenamed("Feature_Time", "dt").write.partitionBy("dt").mode("overwrite").parquet(complex_path_sample_result_base_dir)
    
    # 为每个分区创建成功标志
    for edge_time in edge_times:
        hdfs_create_marker_file(complex_path_sample_result_base_dir + f"/dt={edge_time}", '_SUCCESS')
        
    tmp_str = json.dumps(complex_path_config)
    hdfs_create_marker_file(complex_path_sample_result_base_dir, '_PathConfig', tmp_str)

    tmp_str = json.dumps(edge_tables_feat_info['edge_feature_columns'])
    hdfs_create_marker_file(complex_path_sample_result_base_dir, '_EdgeFeatCols', tmp_str)

    tmp_str = json.dumps(edge_tables_feat_info['edge_feature_comments'])
    hdfs_create_marker_file(complex_path_sample_result_base_dir, '_EdgeFeatComments', tmp_str)
    
    # 返回的结果
    complex_path = {}
    complex_path['data'] = complex_path_df
    complex_path['path_config'] = complex_path_config
    complex_path['edge_feature_columns'] = edge_tables_feat_info['edge_feature_columns']
    complex_path['edge_feature_comments'] = edge_tables_feat_info['edge_feature_comments']
    
    return complex_path

def complex_path_edge_aggregation():
    return

@Time_Costing
def complex_path_node_aggregation(spark, src_node_std_name, tgt_node_std_name, complex_path_name, complex_path, all_nodes_config, agg_funcs, 
                       node_times, result_store_dir, feat_cols_output_type = 'columns'):
    print(f"Node aggregation from node {src_node_std_name} to {tgt_node_std_name} in complex_path {complex_path_name} by functions {agg_funcs} at times {node_times}")
    print(f"output result at {result_store_dir}")
    
    # Check the existence of the aggregation result
    aggregation_result_base_dir = result_store_dir + f"/{complex_path_name}/Node_Aggregation/{src_node_std_name}"
    unfinished_node_times = []
    for node_time in node_times:
        if not hdfs_file_exists(aggregation_result_base_dir + f"/dt={node_time}/_SUCCESS"): 
            unfinished_node_times.append(node_time)
    
    if len(unfinished_node_times) == 0:
        print("The node aggregation result already exists")
        return
    
    ##################################################################################################################################
    # Check the existence of the aggregation result
    node_feature_base_dir = result_store_dir + f"/{complex_path_name}/Node_Feature/{src_node_std_name}"
    finished_node_feature_dirs = []
    for node_time in node_times:
        node_feature_dir = node_feature_base_dir + f"/dt={node_time}"
        if hdfs_file_exists(node_feature_dir + "/_SUCCESS"): 
            finished_node_feature_dirs.append(node_feature_dir)
    
    if len(finished_node_feature_dirs) == len(node_times):
        print("The node feaure result already exists")
        
        src_agg_node_table = {}
        
        src_agg_node_table['data'] = spark.read.option("basePath", node_feature_base_dir).parquet(*finished_node_feature_dirs)
        
        tmp_str = hdfs_read_marker_file(node_feature_base_dir, '_FeatCols')
        src_agg_node_table['feature_columns'] = json.loads(tmp_str)
        
        tmp_str = hdfs_read_marker_file(node_feature_base_dir, '_FeatComments')
        src_agg_node_table['feature_comments'] = json.loads(tmp_str)
    else:
        # Get the corresponding node type
        for edge_table_config in complex_path['path_config']:
            if 'tail_node_std_name' in edge_table_config and edge_table_config['tail_node_std_name'] == src_node_std_name:
                agg_node_type = edge_table_config['tail_node_type']
                break
            elif 'tail_node_std_names' in edge_table_config and src_node_std_name in edge_table_config['tail_node_std_names']:
                agg_node_index = edge_table_config['tail_node_std_names'].index(src_node_std_name)
                agg_node_type = edge_table_config['tail_node_types'][agg_node_index]
                break

        # Only keep the target node columns and the time column
        target_id_columns = [tgt_node_std_name, src_node_std_name, 'Feature_Time']

        # Use the agg_node as the target node
        src_agg_node = {}
        src_agg_node['data'] = complex_path['data'].select(target_id_columns).persist(DEFAULT_STORAGE_LEVEL)
#         src_agg_node['data'] = src_agg_node['data'].repartition(src_node_std_name, "Feature_Time").persist(DEFAULT_STORAGE_LEVEL)
        src_agg_node['node_column'] = src_node_std_name

        # Get the feature of the agg_node
        src_agg_node_table = read_list_node_tables(spark, agg_node_type, all_nodes_config[agg_node_type], node_times, 
                                     target_node = src_agg_node, feat_cols_output_type = 'columns')

        print("Join node tables to the complex-path")
        
        # Join node feature to table
        src_agg_node_table['data'] = src_agg_node_table['data'].withColumnRenamed("Feature_Time", "dt")
        
        # Save result
        src_agg_node_table['data'].write.partitionBy("dt").mode("overwrite").parquet(node_feature_base_dir)
        
        tmp_str = json.dumps(src_agg_node_table['feature_columns'])
        hdfs_create_marker_file(node_feature_base_dir, '_FeatCols', tmp_str)
        
        tmp_str = json.dumps(src_agg_node_table['feature_comments'])
        hdfs_create_marker_file(node_feature_base_dir, '_FeatComments', tmp_str)

    ##################################################################################################################################
    # Only keep the target node and the node features
    agg_node_features_df = src_agg_node_table['data'].select([tgt_node_std_name, 'dt'] + src_agg_node_table['feature_columns'])
    
    # Groupby node features to the start node
    aggregation_result = pyspark_feature_aggregation(spark, agg_node_features_df, [tgt_node_std_name, "dt"],
                                     src_agg_node_table['feature_columns'], agg_funcs)
    
    # Transfer feature columns to vector if required
    
    # Store data
    aggregation_result['data'].write.partitionBy("dt").mode("overwrite").parquet(aggregation_result_base_dir)
    
    # 为每个分区创建成功标志
    for node_time in node_times:
        hdfs_create_marker_file(aggregation_result_base_dir + f"/dt={node_time}", '_SUCCESS')
    
    tmp_str = json.dumps(aggregation_result['feature_columns'])
    hdfs_create_marker_file(aggregation_result_base_dir, '_FeatCols', tmp_str)

    tmp_str = json.dumps(aggregation_result['feature_comments'])
    hdfs_create_marker_file(aggregation_result_base_dir, '_FeatComments', tmp_str)
    
    return

@Time_Costing
def complex_path_aggregation(spark, complex_path_name, complex_path_agg_config, all_nodes_config, edge_times, node_times, result_store_dir, 
                    target_node = None, feat_cols_output_type = 'columns'):
    """"
    为复杂路采样结果的邻居点添加特征

    输入：

    返回值：
    """
    print('Do complex-path aggregation for complex-path:', complex_path_name)
    
    ##############################################################################################
    # Sample Neighbor Nodes
    complex_path = complex_path_sampling(spark, complex_path_name, complex_path_agg_config['path_config'], edge_times, result_store_dir, 
                             target_node, feat_cols_output_type = 'columns')
    
    ##############################################################################################
    # Do complex path aggregation for each target edge (which edge？ and which tail node?)
#     for agg_edge_index, agg_edge_tail_node_std_name in enumerate(complex_path_agg_config["aggregation_edges_tail_nodes"]):
#         if 'edge_agg_funcs_list' in complex_path_agg_config:
#             agg_funcs = complex_path_agg_config['edge_agg_funcs_list'][agg_edge_index]
#         elif 'edge_agg_funcs' in complex_path_agg_config:
#             agg_funcs = complex_path_agg_config['edge_agg_funcs']
#         else:
#             agg_funcs = ['AVG', 'SUM', 'MAX', 'MIN', 'COUNT']
            
    ##############################################################################################
    # Do complex path aggregation for each target node
    for agg_node_index, agg_node_std_name in enumerate(complex_path_agg_config["aggregation_nodes"]):
        if 'node_agg_funcs_list' in complex_path_agg_config:
            agg_funcs = complex_path_agg_config['node_agg_funcs_list'][agg_node_index]
        elif 'node_agg_funcs' in complex_path_agg_config:
            agg_funcs = complex_path_agg_config['node_agg_funcs']
        else:
            agg_funcs = ['AVG', 'MAX', 'MIN']
            
        complex_path_node_aggregation(spark, agg_node_std_name, complex_path['path_config'][0]['head_node_std_name'], complex_path_name, 
                            complex_path, all_nodes_config, agg_funcs, node_times, result_store_dir)
    
    # Add a success mark
#     hdfs_create_marker_file(tmp_complex_path_sample_node_feature_dir + '_Info', '_FeatColsComment')

    # Unpersist data
    
    return
        
#     # Get all the node columns
#     tmp_numeric_cols_list, _ = check_numeric_columns(tmp_meta_path_result_rdd.schema)
#     tmp_all_aim_node_column_name_list = [x for x in tmp_all_aim_column_name_list if x not in tmp_numeric_cols_list]

#     print('Complex Path:', Complex_Path_Name, 'All Output Node Columns:', tmp_all_aim_node_column_name_list)
#     ##############################################################################################
#     # Uniform the start column name
#     tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.withColumnRenamed(Aim_Complex_Path_Start_Column, 'Start_Column')
    
#     # Process each node column
#     for tmp_add_feature_column_i in range(0, len(tmp_all_aim_node_column_name_list) - 1):
#         # Node Column to aggregate
#         tmp_add_feature_column_name = tmp_all_aim_node_column_name_list[tmp_add_feature_column_i + 1]
        
#         # Weight Column of this Node Column
#         tmp_add_feature_columns_list = tmp_node_column_to_weight_dict[tmp_add_feature_column_name]
        
#         # Get the target edge info:
#         tmp_sub_meta_path_feature_result_rdd = tmp_meta_path_result_rdd.select(['Start_Column', tmp_add_feature_column_name] + 
#                                                         tmp_add_feature_columns_list)
#         if len(tmp_all_aim_node_column_name_list) > 1:
#             # Groupby edges with the same node and accumulate the weight
#             tmp_sub_meta_path_feature_result_rdd = Groupby_Feature_Table(Spark_Session, tmp_sub_meta_path_feature_result_rdd, 
#                                                      ['Start_Column', tmp_add_feature_column_name], 
#                                                      tmp_add_feature_columns_list, ['SUM'])
            
#             for tmp_numeric_feat_col in tmp_add_feature_columns_list:
#                 # 待优化命名**
#                 tmp_sub_meta_path_feature_result_rdd = tmp_sub_meta_path_feature_result_rdd.withColumnRenamed('SUM_' + tmp_numeric_feat_col, 
#                                                                                tmp_numeric_feat_col)
        
#         tmp_sub_meta_path_feature_result_rdd.persist(DEFAULT_STORAGE_LEVEL)
        
#         ###############################################################################################################################
#         # Check whether to aggregate the corresponding edge weight
#         if tmp_add_feature_column_name in Complex_Path_Config["Target_Edge_Tail_Column_List"]:
#             print('---------------------------------------------------------------------------------------------------------')
#             print('Aggregate the edge features length:', len(tmp_add_feature_columns_list), 'from the column:', tmp_add_feature_column_name)
            
#             ###############################################################################################################################
#             # Define the table name
#             tmp_meta_path_feature_table_name = ('CompP_W_Feat___' + Complex_Path_Name + '___' + tmp_add_feature_column_name + 
#                                     '___Groupby_Result___' + Table_Name_Comment)
#             if len(tmp_meta_path_feature_table_name) > 120:
#                 tmp_meta_path_feature_table_name = tmp_meta_path_feature_table_name[:120]
#                 print('Cut the table name to 120 characters')
#             print('Output Table Name', tmp_meta_path_feature_table_name)
            
#             # Check whether the corresponding file exists
#             tmp_continue_compute_mark = True
#             if Output_Table_Type in ["HDFS_Stage_1", "HDFS_Stage_2"]:
#                 # Path to store hdfs
#                 tmp_output_pyspark_df_store_dir = (Data_In_HDFS_Path + '/ComplexPath_Aggregation/' + tmp_meta_path_feature_table_name + 
#                                         f"/dt={Upload_dt}")
                        
#                 if hdfs_file_exists(tmp_output_pyspark_df_store_dir + '/_SUCCESS'):
#                     print("File existed:", tmp_output_pyspark_df_store_dir)
#                     tmp_continue_compute_mark = False
        
#             ###############################################################################################################################
#             if tmp_continue_compute_mark:
#                 # Aggregate and upload the result
#                 Upload_Aggregated_Feature_and_Summary(Spark_Session, tmp_sub_meta_path_feature_result_rdd, 'Start_Column', Aim_Node_type, 
#                                           tmp_add_feature_columns_list, tmp_add_feature_columns_list, 
#                                           ['AVG', 'SUM', 'MAX', 'MIN', 'COUNT'], tmp_meta_path_feature_table_name, 
#                                           Aim_Relation_Table_dt, Aim_Relation_Table_dt, Output_Table_Type)
                
#                 print("Finish the aggregation edge features of the node:", tmp_add_feature_column_name)
                
#             print("----------------------------------------------------------------------------------------------")
        
#         ###############################################################################################################################
#         # Check whether to aggregate the corresponding node weight
#         if tmp_add_feature_column_name in Complex_Path_Config["Target_Node_Column_List"]:
#             print('Aggregate the node features from column:', tmp_add_feature_column_name)

#             ###############################################################################################################################
#             # Define the table name
#             tmp_meta_path_feature_table_name = ('CompP_N_Feat___' + Complex_Path_Name + '___' + 
#                                      tmp_add_feature_column_name + '___Groupby_Result___' + Table_Name_Comment)
#             if len(tmp_meta_path_feature_table_name) > 120:
#                 tmp_meta_path_feature_table_name = tmp_meta_path_feature_table_name[:120]
#                 print('Cut the table name to 120 characters')
#             print('Output Table Name', tmp_meta_path_feature_table_name)
            
#             # Check whether the corresponding file exists
#             tmp_continue_compute_mark = True
#             if Output_Table_Type in ["HDFS_Stage_1", "HDFS_Stage_2"]:
#                 # Path to store hdfs
#                 tmp_output_pyspark_df_store_dir = (Data_In_HDFS_Path + '/ComplexPath_Aggregation/' + tmp_meta_path_feature_table_name + 
#                                         f"/dt={Upload_dt}")
                        
#                 if hdfs_file_exists(tmp_output_pyspark_df_store_dir + '/_SUCCESS'):
#                     print("File existed:", tmp_output_pyspark_df_store_dir)
#                     tmp_continue_compute_mark = False
            
#             ##############################################################################################
#             if tmp_continue_compute_mark:
#                 # Temporary_Data_Store_Path(得加上Aim_Feature_Table_dt **)
#                 tmp_complex_path_sample_node_feature_dir = (Data_In_HDFS_Path + '/Temporary_Data/Sample_Node_Feat/' + Complex_Path_Name + '___' + 
#                                               Table_Name_Comment + f"/edge_dt={Aim_Relation_Table_dt}" + 
#                                               f"/feat_dt={Aim_Feature_Table_dt}")
                
#                 # Temporary groupby data store path
#                 tmp_groupby_result_dir = (Data_In_HDFS_Path + '/Temporary_Data/Groupby_Result/' + tmp_meta_path_feature_table_name + 
#                                   f"/dt={Upload_dt}")
                
#                 if not hdfs_file_exists(tmp_groupby_result_dir + '/_SUCCESS'):
#                     # Check the existence of sample node feature data
#                     if not hdfs_file_exists(tmp_complex_path_sample_node_feature_dir + '/_SUCCESS'):

#                         #########################################################################################################
#                         # Get the node type
#                         tmp_add_feature_node_type = tmp_column_name_to_class_dict[tmp_add_feature_column_name]
#                         print('Node type:', tmp_add_feature_node_type)

#                         # Only keep the target columns
# #                         tmp_sub_meta_path_result_rdd = tmp_meta_path_result_rdd.select(['Start_Column', tmp_add_feature_column_name])
#                         tmp_sub_meta_path_result_rdd = tmp_sub_meta_path_feature_result_rdd.select(['Start_Column', 
#                                                                            tmp_add_feature_column_name])
                        
#                         # Change the column name for src node
#                         tmp_aim_path_result_rdd = tmp_sub_meta_path_result_rdd.withColumnRenamed(tmp_add_feature_column_name, 
#                                                                          tmp_add_feature_node_type + '_UID')

#                         # Get the distinct src node
#                         tmp_UID_for_add_feature = tmp_aim_path_result_rdd.select([tmp_add_feature_node_type + '_UID']).distinct()

#                         tmp_UID_for_add_feature = tmp_UID_for_add_feature.persist(DEFAULT_STORAGE_LEVEL)

#                         #########################################################################################################
#                         # Add Node Features
#                         tmp_all_useful_feature_cols_list = []
#                         tmp_all_useful_feature_cols_comments_list = []

#                         for tmp_feature_table_Info_i in range(0, len(Feature_Dataset_Config_dict[tmp_add_feature_node_type]['Feature_Data_List'])):
#                             tmp_add_feature_start_time = datetime.now()

#                             tmp_feature_table_Info_dict = Feature_Dataset_Config_dict[tmp_add_feature_node_type]['Feature_Data_List'][tmp_feature_table_Info_i]

#                             tmp_sub_feature_table_rdd = Read_Feature_Data_from_Node_Table(Spark_Session, tmp_UID_for_add_feature, 
#                                                                       tmp_feature_table_Info_dict, 
#                                                                       Aim_Feature_Table_dt, tmp_add_feature_node_type,
#                                                                       tmp_all_useful_feature_cols_list,
#                                                                       tmp_all_useful_feature_cols_comments_list)

#                             # If just start, then use tmp_sub_feature_table_rdd or join with the past result
#                             if tmp_feature_table_Info_i == 0:
#                                 tmp_all_feature_table_rdd = tmp_sub_feature_table_rdd

#                             else:
#                                 # 因为已经保证ID相同，所以只用left join就行
#                                 tmp_all_feature_table_rdd = tmp_all_feature_table_rdd.join(tmp_sub_feature_table_rdd, 
#                                                                         tmp_add_feature_node_type + '_UID', 'left')

#                                 tmp_all_feature_table_rdd = tmp_all_feature_table_rdd.repartition(tmp_add_feature_node_type + '_UID')


#                         tmp_all_feature_table_rdd = tmp_all_feature_table_rdd.fillna(0)

#                         tmp_all_feature_table_rdd = tmp_all_feature_table_rdd.persist(DEFAULT_STORAGE_LEVEL)

#                         # Join all the feature to the path
#                         tmp_sub_meta_path_feature_result_rdd = tmp_aim_path_result_rdd.join(tmp_all_feature_table_rdd, 
#                                                                       tmp_add_feature_node_type + '_UID', 'left')

#                         # Delete the UID column of the src node
#                         tmp_sub_meta_path_feature_result_rdd = tmp_sub_meta_path_feature_result_rdd.drop(tmp_add_feature_node_type + '_UID')

#                         tmp_all_useful_feature_cols_list_str = json.dumps(tmp_all_useful_feature_cols_list)
#                         hdfs_create_marker_file(tmp_complex_path_sample_node_feature_dir+ '_Info', '_FeatCols', tmp_all_useful_feature_cols_list_str)

#                         tmp_all_useful_feature_cols_comments_list_str = json.dumps(tmp_all_useful_feature_cols_comments_list)
#                         hdfs_create_marker_file(tmp_complex_path_sample_node_feature_dir + '_Info', '_FeatColsComment', 
#                                         tmp_all_useful_feature_cols_comments_list_str)

#                         # Keep the present result
#                         tmp_sub_meta_path_feature_result_rdd = tmp_sub_meta_path_feature_result_rdd.persist(DEFAULT_STORAGE_LEVEL)

#                         tmp_sub_meta_path_feature_result_rdd.write.mode("overwrite").parquet(tmp_complex_path_sample_node_feature_dir)
                        
#                         # Nodes number with features
#                         tmp_UID_for_add_feature_counts = tmp_UID_for_add_feature.count()
#                         print('Nodes count to add features:', tmp_UID_for_add_feature_counts)
                        
#                         tmp_UID_for_add_feature.unpersist()
#                     else:
#                         print("Sampled nodes feature file existed:", tmp_groupby_result_dir)

#                         tmp_sub_meta_path_feature_result_rdd = Spark_Session.read.parquet(tmp_complex_path_sample_node_feature_dir)

#                         # Keep the present result
#                         tmp_sub_meta_path_feature_result_rdd = tmp_sub_meta_path_feature_result_rdd.persist(DEFAULT_STORAGE_LEVEL)

#                         tmp_all_useful_feature_cols_list_str = hdfs_read_marker_file(tmp_complex_path_sample_node_feature_dir + '_Info', '_FeatCols')
#                         tmp_all_useful_feature_cols_list = json.loads(tmp_all_useful_feature_cols_list_str)

#                         tmp_all_useful_feature_cols_comments_list = hdfs_read_marker_file(tmp_complex_path_sample_node_feature_dir + '_Info', 
#                                                                     '_FeatColsComment')
#                         tmp_all_useful_feature_cols_comments_list = json.loads(tmp_all_useful_feature_cols_comments_list)

#                         print("All feature columns length:", len(tmp_all_useful_feature_cols_list))
                        
# #                     # Join all the feature to the path
# #                     tmp_sub_meta_path_feature_result_rdd = tmp_aim_path_result_rdd.join(tmp_all_feature_table_rdd, 
# #                                                                   tmp_add_feature_node_type + '_UID', 'left')

#                     ##############################################################################################
#                     # Aggregate and Upload the result
#                     Upload_Aggregated_Feature_and_Summary(Spark_Session, tmp_sub_meta_path_feature_result_rdd, 'Start_Column', Aim_Node_type, 
#                                               tmp_all_useful_feature_cols_list, tmp_all_useful_feature_cols_comments_list, 
#                                               Groupby_Type_List, tmp_meta_path_feature_table_name, 
#                                               Aim_Relation_Table_dt, Aim_Feature_Table_dt, Output_Table_Type)
#                 else:
#                     # Aggregate and Upload the result
#                     Upload_Aggregated_Feature_and_Summary(Spark_Session, None, 'Start_Column', Aim_Node_type, 
#                                               None, None, Groupby_Type_List, tmp_meta_path_feature_table_name, 
#                                               Aim_Relation_Table_dt, Aim_Feature_Table_dt, Output_Table_Type)
                
#                 print("Finish the aggregation node features of the node:", tmp_output_pyspark_df_store_dir)
            
#             print("----------------------------------------------------------------------------------------------")
        
#         tmp_sub_meta_path_feature_result_rdd.unpersist()
        
#     ##############################################################################################
#     tmp_meta_path_rows = tmp_meta_path_result_rdd.count()
#     print('Complex-Path Rows:', tmp_meta_path_rows)
    
#     # Get the Start Column
#     tmp_meta_path_result_start_column_rdd = tmp_meta_path_result_rdd.select('Start_Column')

#     # Drop duplicates
#     tmp_meta_path_result_start_node_rdd = tmp_meta_path_result_start_column_rdd.distinct()
    
#     # Nodes number in the start column
#     tmp_start_node_counts = tmp_meta_path_result_start_node_rdd.count()
#     print('Start Column Nodes number:', tmp_start_node_counts)
    
#     tmp_meta_path_result_rdd = tmp_meta_path_result_rdd.unpersist()
    
#     ##############################################################################################
#     print('Finish the aggregation for complex-path:', Complex_Path_Name)
#     print('##########################################################################################')
    
#     return

"""

"""
def Upload_Aggregated_Feature_and_Summary(Spark_Session, Path_Feature_Table_df, Grouping_Column, Aim_Node_type, Aggregation_Columns, 
                            Aggregation_Columns_Comment, Groupby_Type_List, Upload_Table_Name, Aim_Relation_Table_dt, 
                            Aim_Feature_Table_dt, Output_Table_Type):
    print("----------------------------------------------------------------------------")
    print("Start Grouping")
    
    groupby_upload_start_time = datetime.now()
    
    # Upload dt
    Upload_dt = Aim_Relation_Table_dt
    if Aim_Relation_Table_dt != Aim_Feature_Table_dt:
        Upload_dt = Upload_dt + '|' + Aim_Feature_Table_dt
    
    ###############################################################################################################################
    
    # 聚合特征数据，并返回聚合后的结果，或直接读取已存在的结果
    def Get_or_Read_Groupby_Result(tmp_groupby_result_dir, tmp_target_feat_df, tmp_target_grouping_column, tmp_target_feat_columns):
        
        # Check whether the corresponding file exists
        if not hdfs_file_exists(tmp_groupby_result_dir + '/_SUCCESS'):
            tmp_target_columns_for_groupby = [tmp_target_grouping_column] + tmp_target_feat_columns

            tmp_target_sub_feat_df = tmp_target_feat_df.select(tmp_target_columns_for_groupby)


            # Get the groupby result
            tmp_groupby_result_rdd = Groupby_Feature_Table(Spark_Session, tmp_target_sub_feat_df, tmp_target_grouping_column,
                                            tmp_target_feat_columns, Groupby_Type_List)

            ######################################################################################################################
            # Get all the column names
            tmp_groupby_feature_comment_list = [Aim_Node_type + '_UID']
            for tmp_groupby_type in Groupby_Type_List:
                if tmp_groupby_type in ['AVG', 'SUM', 'MAX', 'MIN']:
                    tmp_groupby_feature_comment_list = (tmp_groupby_feature_comment_list + 
                                             [tmp_groupby_type + '_' + x for x in Aggregation_Columns_Comment])
                elif tmp_groupby_type == 'COUNT':
                    tmp_groupby_feature_comment_list = tmp_groupby_feature_comment_list + ["该关系对应邻接点个数"]
                else:
                    continue
            
            ######################################################################################################################
            tmp_groupby_feature_comment_list_str = json.dumps(tmp_groupby_feature_comment_list)
            hdfs_create_marker_file(tmp_groupby_result_dir + '_Info', '_FeatColsComment', tmp_groupby_feature_comment_list_str)
            
            tmp_groupby_result_rdd.persist(DEFAULT_STORAGE_LEVEL)
            tmp_groupby_result_rdd.write.mode("overwrite").parquet(tmp_groupby_result_dir)

            print("Finish grouping")
        else:
            print("Grouped file exists")

            tmp_groupby_result_rdd = Spark_Session.read.parquet(tmp_groupby_result_dir)

            tmp_groupby_feature_comment_list_str = hdfs_read_marker_file(tmp_groupby_result_dir + '_Info', '_FeatColsComment')
            tmp_groupby_feature_comment_list = json.loads(tmp_groupby_feature_comment_list_str)
            
            tmp_groupby_result_rdd.persist(DEFAULT_STORAGE_LEVEL)
        
        return tmp_groupby_result_rdd, tmp_groupby_feature_comment_list
    
    ###############################################################################################################################
    
    def Preprocess_and_Upload_Data_Based_on_Output_Type(tmp_groupby_result_rdd, tmp_groupby_feature_comment_list, Upload_Table_Extra_comment = ''):
        tmp_groupby_feature_name_list = list(tmp_groupby_result_rdd.columns)[1:]
        
        print('The number of raw features to upload:', len(tmp_groupby_feature_name_list))
    
        ###############################################################################################################################
        if Output_Table_Type == 'HDFS_Stage_1':
            # Path to store hdfs
            tmp_output_pyspark_df_store_dir = (Data_In_HDFS_Path + '/ComplexPath_Aggregation/' + Upload_Table_Name + Upload_Table_Extra_comment
                                    + f"/dt={Upload_dt}")

            # Check whether the corresponding file exists
            if not hdfs_file_exists(tmp_output_pyspark_df_store_dir + '/_SUCCESS'):
            
                print('Start writing to:', tmp_output_pyspark_df_store_dir)
                
                # Store the feature column comment
                tmp_groupby_feature_comment_list_str = json.dumps(tmp_groupby_feature_comment_list)
                hdfs_create_marker_file(tmp_output_pyspark_df_store_dir + '_Info', '_Feature_Comment', 
                                tmp_groupby_feature_comment_list_str)
                
                # Store to target dir
                tmp_groupby_result_rdd.write.mode("overwrite").parquet(tmp_output_pyspark_df_store_dir)

            else:
                print("Target file exists:", tmp_output_pyspark_df_store_dir)
            
        elif Output_Table_Type == 'HDFS_Stage_2':
            # Path to store hdfs
            tmp_output_pyspark_df_store_dir = (Data_In_HDFS_Path + '/ComplexPath_Aggregation/' + Upload_Table_Name + Upload_Table_Extra_comment 
                                    + f"/dt={Upload_dt}")

            # Check whether the corresponding file exists
            if not hdfs_file_exists(tmp_output_pyspark_df_store_dir + '/_SUCCESS'):
#                 # Preprocessing
#                 tmp_agg_feature_for_upload = Preprocess_Numerical_Data(tmp_groupby_result_rdd, ['Raw', 'Normalized', 'Standard'], 
#                                                      [Grouping_Column], tmp_groupby_feature_name_list)
                
                # Preprocessing
                tmp_agg_feature_for_upload = Preprocess_Numerical_Data(tmp_groupby_result_rdd, ['Raw', 'Normalized'], 
                                                 [Grouping_Column], tmp_groupby_feature_name_list)
            
                print('Start writing to:', tmp_output_pyspark_df_store_dir)

                # Store the feature column comment
                tmp_groupby_feature_comment_list_str = json.dumps(tmp_groupby_feature_comment_list)
                hdfs_create_marker_file(tmp_output_pyspark_df_store_dir + '_Info', '_Feature_Comment', 
                                tmp_groupby_feature_comment_list_str)
                
                # Store to target dir
                tmp_agg_feature_for_upload.write.mode("overwrite").parquet(tmp_output_pyspark_df_store_dir)

            else:
                print("Target file exists:", tmp_output_pyspark_df_store_dir)
                
        elif Output_Table_Type == 'Online':
            # Upload batch
            Upload_batch_count = math.ceil((len(tmp_groupby_feature_name_list) * tmp_groupby_result_rdd.count()) / 10000000000)

            # Upload data
            Upload_RDD_Data_to_Database(Spark_Session, 'tmp.tmp_' + Upload_Table_Name + Upload_Table_Extra_comment, tmp_groupby_result_rdd, 
                               Upload_dt, Set_Table_Columns_Comment_List = tmp_groupby_feature_comment_list, 
                               batch_count = Upload_batch_count)
    
        return
    
    ###############################################################################################################################
    # *待优化，要在配置文件中添加最大列数及应该采样的分区数
    tmp_max_agg_columns_len = 300
    
#     # Repartition based on the data size
#     Path_Feature_Table_df = Path_Feature_Table_df.repartition(10000, Grouping_Column)
    
    # 先确定是否要分分批运算
    if len(Aggregation_Columns) > tmp_max_agg_columns_len:
        
        # Persist data
        Path_Feature_Table_df.persist(DEFAULT_STORAGE_LEVEL)
        
        for tmp_agg_column_start in range(0, len(Aggregation_Columns), tmp_max_agg_columns_len):
            tmp_agg_column_end = tmp_agg_column_start + tmp_max_agg_columns_len
            if tmp_agg_column_end > len(Aggregation_Columns):
                tmp_agg_column_end = len(Aggregation_Columns)

            # Path to store sub groupby result
            tmp_sub_groupby_result_dir = (Data_In_HDFS_Path + '/Temporary_Data/Sub_Groupby_Result_' + 
                                f'{tmp_agg_column_start}_{tmp_agg_column_end}/' + Upload_Table_Name + f"/dt={Upload_dt}")
                
            print(f"Groupby features from {tmp_agg_column_start} to {tmp_agg_column_end}. Output file:", tmp_sub_groupby_result_dir)

            tmp_target_feat_columns = Aggregation_Columns[tmp_agg_column_start:tmp_agg_column_end]
            
            tmp_groupby_result_rdd, tmp_groupby_result_comments = Get_or_Read_Groupby_Result(tmp_sub_groupby_result_dir, Path_Feature_Table_df, 
                                                                  Grouping_Column, tmp_target_feat_columns)
            
            if 'COUNT' in Groupby_Type_List and tmp_agg_column_start != 0:
                tmp_groupby_result_rdd = tmp_groupby_result_rdd.drop('Groupby_COUNT')
            
            tmp_groupby_result_rdd = tmp_groupby_result_rdd.repartition(Grouping_Column)
            
            Preprocess_and_Upload_Data_Based_on_Output_Type(tmp_groupby_result_rdd, tmp_groupby_result_comments, 
                                             f'_{tmp_agg_column_start}_{tmp_agg_column_end}')
                
        Path_Feature_Table_df.unpersist()
    else:
        # Path to store hdfs
        tmp_groupby_result_dir = (Data_In_HDFS_Path + '/Temporary_Data/Groupby_Result/' + Upload_Table_Name + f"/dt={Upload_dt}")
        
        tmp_groupby_result_rdd, tmp_groupby_result_comments = Get_or_Read_Groupby_Result(tmp_groupby_result_dir, Path_Feature_Table_df, 
                                                              Grouping_Column, Aggregation_Columns)
        
        Preprocess_and_Upload_Data_Based_on_Output_Type(tmp_groupby_result_rdd, tmp_groupby_result_comments)
    ###############################################################################################################################
    
    groupby_upload_end_time = datetime.now()
    
    print(datetime.now(), 'Upload Time cost:', groupby_upload_end_time - groupby_upload_start_time)
    print("----------------------------------------------------------------------------------------------")    
    
    return