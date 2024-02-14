from Utils.Pyspark_utils import hdfs_file_exists, hdfs_list_files, hdfs_read_txt_file, hdfs_read_marker_file
from Utils.Pyspark_utils import hdfs_create_marker_file, hdfs_create_dir, Groupby_Feature_Table
from Utils.Pyspark_utils import check_numeric_columns, Preprocess_Numerical_Data, DEFAULT_STORAGE_LEVEL
from Utils.Pyspark_utils import Spark_Random_N_Sample, Spark_Top_N_Sample, Spark_Threshold_N_Sample
from Utils.Pyspark_utils import Data_In_HDFS_Path, estimate_partitions, sanitize_column_names
from Utils.Complex_Path_Basic_PySpark import read_edge_table

from Utils.utils import mkdir
from Utils.Decorator import Time_Costing

from pyspark.sql import Row
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, lit, broadcast, split, substring, expr, concat
from pyspark.sql.functions import max, min, sum, count, datediff, to_date, when
from pyspark.sql.types import DoubleType
from pyspark.sql.types import *
from pyspark.sql.functions import udf, array
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

import os
import re
import json
import numpy as np
import pandas as pd
from functools import reduce

def read_existing_edges(spark, new_paths_dir):
    """
    读取已存在的各条new_path
    """
    # 存储该跳获取的新路径的信息
    new_paths_list = []

    if not hdfs_file_exists(new_paths_dir):
        print("该跳还未有任何路径被生成")
        return new_paths_list
    
    # 获取对应跳数
    tmp_hop = int(new_paths_dir.split('/')[-1])

    # 获取该跳下全部的路径文件夹
    path_dirs = hdfs_list_files(new_paths_dir)
    path_dirs = [tmp_path_dir for tmp_path_dir in path_dirs if tmp_path_dir.split('/')[-1].isdigit()]
    print("Existing path dirs:", path_dirs)
    
    if len(path_dirs) > 0:
        # 保证处理顺序和文件夹名称序号一致
        def extract_number(tmp_path_dir):
            return int(tmp_path_dir.split('/')[-1])
        path_dirs = sorted(path_dirs, key = extract_number)

        # 只保留有path_info标记已完成处理的最后路径
        tmp_finished_path_dir_i = 0
        for tmp_i, tmp_path_dir in enumerate(path_dirs):
            if not hdfs_file_exists(tmp_path_dir + "/_PATH_CONFIG"):
                break
            else:
                tmp_finished_path_dir_i = tmp_i + 1
        path_dirs = path_dirs[:tmp_finished_path_dir_i]
        print("Finished path dirs:", path_dirs)
        
        if tmp_finished_path_dir_i > 0:
            print("已处理完路径数:", len(path_dirs), "处理到:", path_dirs[-1])

            # 依次读取各个路径的采样结果,具体信息以及路径名
            for tmp_path_dir in path_dirs:
                tmp_new_path_index = int(tmp_path_dir.split('/')[-1])
                tmp_path_name = f'{tmp_hop}_{tmp_new_path_index}'

                tmp_path_data = spark.read.parquet(tmp_path_dir + '/Path_Data')

                tmp_path_info_str = hdfs_read_marker_file(tmp_path_dir, '_PATH_CONFIG')
                tmp_path_info = json.loads(tmp_path_info_str)

                tmp_path_data = tmp_path_data.repartition(tmp_path_info[-1]['tail_node_std_name'])

                new_paths_list.append({"data": tmp_path_data, "path_config": tmp_path_info, "path_name": tmp_path_name})

                # 如果有之前的匹配结果，则也打印一下
                if hdfs_file_exists(tmp_path_dir + '/_PathCoverInfo'):
                    cover_summary_str = hdfs_read_marker_file(tmp_path_dir, '_PathCoverInfo')
                    cover_summary = json.loads(cover_summary_str)

                    print("path info:", tmp_path_info)
                    print("cover summary:", cover_summary)
    
    print(f"该跳已生成{len(new_paths_list)}条路径")
    return new_paths_list

def get_add_edge_config(k_hop_sample_info, edge_name, edge_start_node, hop_k):
    """
    将原始的边在k_hop采样中对应的信息，转化为符合复杂路的path信息
    """
    connect_node_name = k_hop_sample_info[edge_name][edge_start_node + '_Node_name']
    connect_node_type = k_hop_sample_info[edge_name][edge_start_node + '_Node_class']

    edge_end_node = "Tail"
    if edge_start_node == "Tail":
        edge_end_node = "Head"
    add_node_name = k_hop_sample_info[edge_name][edge_end_node + '_Node_name']
    add_node_type = k_hop_sample_info[edge_name][edge_end_node + '_Node_class']

    # 获取这条边对应的配置信息
    add_edge_config = {}
    add_edge_config["head_node_raw_name"] = connect_node_name
    add_edge_config["head_node_std_name"] = f'Node_{hop_k}'
    add_edge_config["head_node_type"] = connect_node_type
    add_edge_config["edge_table_name"] = k_hop_sample_info[edge_name]['Relation_Data']
    if 'Relation_Data_Source' in k_hop_sample_info[edge_name]:
        add_edge_config["edge_table_source"] = k_hop_sample_info[edge_name]['Relation_Data_Source']
    add_edge_config["tail_node_raw_name"] = add_node_name
    add_edge_config["tail_node_std_name"] = f'Node_{hop_k + 1}'
    add_edge_config["tail_node_type"] = add_node_type
    
    if edge_start_node + '_Edge_Feature_Limits' in k_hop_sample_info[edge_name]:
        add_edge_config["Edge_Feature_Limits"] = k_hop_sample_info[edge_name][edge_start_node + '_Edge_Feature_Limits'].copy()
    if edge_start_node + '_Node_Feature_Limits' in k_hop_sample_info[edge_name]:
        add_edge_config["Node_Feature_Limits"] = k_hop_sample_info[edge_name][edge_start_node + '_Node_Feature_Limits'].copy()
    if edge_start_node + '_Edge_Neighbor_Limits' in k_hop_sample_info[edge_name]:
        add_edge_config["Edge_Neighbor_Limits"] = k_hop_sample_info[edge_name][edge_start_node + '_Edge_Neighbor_Limits'].copy()
        
    return add_edge_config

@Time_Costing
def generate_new_path(spark, hop_k, seed_path_index, seed_path, add_edge_name, add_edge_config, add_edge_times, new_paths_dir, new_path_index, 
               target_pair = None):
    """
    针对指定的seed_path和add_edge生成新的path
    """
    result_dir = new_paths_dir + f'/{new_path_index}'
    
    # 查看该跳对应的第i个seed_path和该edge对应的结果是否已存在
    new_path_name = f'{hop_k}_{new_path_index}'
    path_success_mark_file = f"_SUCCESS_Marker_{new_path_name}"
    
    if not hdfs_file_exists(result_dir + '/' + path_success_mark_file):
        # 获得新path的配置信息
        if "head_node_std_name" in seed_path['path_config'][0]:
            new_path_config = seed_path['path_config'].copy()
        else:
            new_path_config = []
        new_path_config.append(add_edge_config)
        print('Start generating new path, the corresponding config is:', new_path_config)
    
        # 获取seed_path中要用来连接的点对应的全部值
        path_target_node = {}
        path_join_id_columns = [add_edge_config["head_node_std_name"], 'Feature_Time']
        path_target_node['data'] = seed_path["data"].select(path_join_id_columns).distinct()
        path_target_node['node_column'] = add_edge_config["head_node_type"]
        
        # 获得对应的edge
        add_edge = read_edge_table(spark, add_edge_config, add_edge_times, path_target_node)
        
        # 获得id列以及特征列
        edge_id_columns = [add_edge_config['head_node_std_name'], add_edge_config['tail_node_std_name'], 'Feature_Time']
        edge_feature_columns, edge_feature_comments = check_numeric_columns(add_edge['data'].schema)
        print('The feature columns of the add edge:', len(edge_feature_columns))
    
        # 给全部边的特征列加上后缀，防止重名
        for edge_feature_column in edge_feature_columns:
            add_edge['data'] = add_edge['data'].withColumnRenamed(edge_feature_column, 
                                                f'Edge_Feat_{edge_feature_column}_to_' + add_edge_config['tail_node_std_name'])

        # 和seed_path进行连接
        new_path_df = seed_path["data"].join(add_edge['data'], on = path_join_id_columns, how = 'left')
    
        # 获得之前的有重复节点类型的列
        duplicate_node_columns = []
        for edge_config in new_path_config:
            if edge_config['head_node_type'] == new_path_config[-1]['tail_node_type']:
                duplicate_node_columns.append(edge_config['head_node_std_name'])
            
        # 如果之前有相同类型的节点列，则去一下重，保证同一个path中不会有重复节点
        if len(duplicate_node_columns) > 0:
            condition = None
            for column in duplicate_node_columns:
                if condition is None:
                    condition = col(new_path_config[-1]['tail_node_std_name']) != col(column)
                else:
                    condition = condition & (col(new_path_config[-1]['tail_node_std_name']) != col(column))
            new_path_df = new_path_df.filter(condition)
    
        # persist结果
        new_path_df.persist(DEFAULT_STORAGE_LEVEL)

        # 显示新路径行数
        new_path_count = new_path_df.count()
        print('The count of new paths:', new_path_count)
        
        # 获得最优分区数
        best_partitions = estimate_partitions(new_path_df, approximate_row_count = new_path_count)
        print('The best partitions for new path:', best_partitions)
        
        # 保存新路径结果
        new_path_df.coalesce(best_partitions).write.mode("overwrite").parquet(result_dir + '/Path_Data')
        print('完成该path的生成')

        # 记录新路径的全部信息
        new_path = {"data": new_path_df, "path_config": new_path_config, "path_name": new_path_name}

        # 如果有目标pair，且终点类型和当前path一致，则进行pair的对比
        if target_pair is not None and new_path_config[-1]['tail_node_type'] == target_pair['End_Node_Type']:
            Pyspark_K_Hop_Path_Pairing(new_path, target_pair, result_dir)

        # 保存新路径信息，并以此标志已完成处理
        new_path_config_str = json.dumps(new_path_config)
        hdfs_create_marker_file(result_dir, '_PATH_CONFIG', new_path_config_str)
    else:
        new_path_config_str = hdfs_read_marker_file(result_dir, '_PATH_CONFIG')
        new_path_config = json.loads(new_path_config_str)
        
        new_path_df = spark.read.parquet(result_dir + '/Path_Data')
        new_path_df = new_path_df.repartition(new_path_config[-1]['tail_node_std_name'])

        new_path = {"data": new_path_df, "path_config": new_path_config, "path_name": new_path_name}

        # 如果有之前的匹配结果，则也打印一下
        if hdfs_file_exists(result_dir + '/_PathCoverInfo'):
            cover_summary_str = hdfs_read_marker_file(result_dir, '_PathCoverInfo')
            cover_summary = json.loads(cover_summary_str)

            print("path info:", new_path_config)
            print("cover summary:", cover_summary)
        
    return new_path


"""
作用：
    针对目标节点和subgraph配置文件，获取subgraph，并将运算结果（采样后的关系）保存在指定文件夹下（运算前会先检验文件是否已存在，若已存在则会跳过生成），同时会更新节点表中的节点信息，将新增节点添加到对应的节点文件中

输入：
    Spark_Session：pyspark接口
    tmp_aim_entity_info_dict：目标节点的相关信息（包含全量的目标节点，以及目标节点的类型和对应日期）
    Subgraph_Config_dict：subgraph配置文件
    tmp_subgraph_hop: 采样多少跳的subgraph
    tmp_subgraph_output_dir：subgraph输出文件夹
    tmp_node_output_dir：涉及到的节点输出文件夹

返回值：
    无返回值，运算结果都会保存在tmp_subgraph_output_dir和tmp_node_output_dir这两个文件夹下
"""
def k_hop_sampling(spark, target_node, k_hop_sampling_config, max_hop_k, result_store_dir, target_pair = None):
    if hdfs_file_exists(result_store_dir + f"/{max_hop_k-1}/_SUCCESS_Marker"):
        print("已完成目标节点全部k-hop路径的采样")
        return
    
    # 先获取目标起始节点及对应的信息
    target_node_df = target_node['Target_Node_df']
    target_node_type = target_node['Target_Node_Type']
    target_node_column = target_node['Target_Node_UID_name']
    edge_times = target_node["Feature_Times"]
    
    print('将目标节点设为第0跳的seed_nodes')
    
    # 将节点列名改成Node_0
    tmp_target_node_df = target_node_df.withColumnRenamed(target_node_column, 'Node_0')
    
    seed_nodes_list = [{"data": target_node_df, "node_type": target_node_type, "node_column": 'Node_0'}]
    
    ##############################################################################################    
    # 依次处理各跳信息
    print("#####" * 20)
    for hop_k in range(max_hop_k):
        print(f'开始处理第{hop_k}跳的关系')
        
        # 该跳对应节点信息的存储文件夹
        new_hop_result_dir = result_store_dir + f"/{hop_k}"
        
        # 获取该跳已存在的新路径的信息
        new_paths_list = read_existing_paths(spark, new_hop_result_dir)
                        
        ##############################################################################################
        # 检查是否已经完成对该跳的处理
        if hdfs_file_exists(new_hop_result_dir + "/_SUCCESS_Marker"):
            print(f"已完成第{hop_k}跳的全部路径的处理")
            
            # 释放seed_paths
            for seed_path in seed_paths_list:
                seed_path["data"].unpersist()

            # 将该跳对应的path作为下一跳的seed_path
            seed_paths_list = new_paths_list
            print("#####" * 20)
            continue
            
        # 依次处理各个的seed_path
        for seed_path_index, seed_path in enumerate(seed_paths_list):
            # 查看是否已完成该seed_path的处理
            if hdfs_file_exists(new_hop_result_dir + f'/_SUCCESS_SEED_PATH_{seed_path_index}'):
                print(f'跳过,已完成第{seed_path_index}个seed_path能生成的全部路径的处理')
                continue

            print(f"开始处理第{seed_path_index}个seed_path:", seed_path["path_config"])

            # 遍历全部关系
            for add_edge_name in k_hop_sampling_config:
                # 依次处理两种方向
                for edge_start_node in ["Head", "Tail"]:
                    # 获取该起始点对应的边的配置信息
                    add_edge_config = get_add_edge_config(k_hop_sampling_config, add_edge_name, edge_start_node, hop_k)

                    print(f'尝试从{edge_start_node}方向为该path添加上边{add_edge_name}:', add_edge_config)
                    
                    # 检查该配置是否需要进行采样（点的类型是否符合要求，以及配置里是否要求进行采样）
                    if seed_path['path_config'][-1]['tail_node_type'] != add_edge_config["head_node_type"]:
                        print('edge and seed path not match')
                        continue

                    if "Edge_Neighbor_Limits" in add_edge_config and add_edge_config["Edge_Neighbor_Limits"]["Max_Num"] <= 0:
                        print('Skip this edge, since neighbor number limits to 0')
                        continue

                    # 若当前已是最后一跳，且终点不为目标pair的终点类型，则跳过
                    if hop_k == (max_hop_k - 1) and target_pair is not None:
                        if target_pair['End_Node_Type'] != add_edge_config['tail_node_type']:
                            print('最后一跳的终止节点不符合，故跳过')
                            continue

                    # 开始生成new_path
                    new_path = generate_new_path(spark, hop_k, seed_path_index, seed_path, add_edge_name, add_edge_config, 
                                        edge_times, new_hop_result_dir, len(new_paths_list), target_pair)

                    # 记录结果
                    new_paths_list.append(new_path)

            hdfs_create_marker_file(new_hop_result_dir, f'_SUCCESS_SEED_PATH_{seed_path_index}')
            print(f'完成第{seed_path_index}个seed_path能生成的全部路径的处理')

        hdfs_create_marker_file(new_hop_result_dir)
        print(f"完成第{hop_k}跳能新增的全部路径的处理")
      
        ##############################################################################################
        # 释放seed_paths
        for seed_path in seed_paths_list:
            seed_path["data"].unpersist()
        
        # 将该跳对应的path作为下一跳的seed_path
        seed_paths_list = new_paths_list
        print("#####" * 20)
    
    print("已完成目标节点的k-hop路径的全部采样")
    
    return