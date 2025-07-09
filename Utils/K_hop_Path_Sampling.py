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

@Time_Costing
def Pyspark_K_Hop_Path_Pairing(target_path, target_pair, result_dir):
    """
    匹配路径和目标对，显示匹配结果，并存储结果
    """
    # 目标path的信息
    target_path_df = target_path["data"]
    target_path_info = target_path["path_config"]
    target_path_name = target_path["path_name"]
    
    # 目标pair信息
    target_pair_df = target_pair['Pair_Data']
    #################################################################################################################################
    
    # 如果终点类型和设定的不一致，则跳过
    if target_path_info[-1]['tail_node_type'] != target_pair['End_Node_Type']:
        print(f"该路径终止点为{tmp_path_info[-1]['tail_node_type']},目标pair的终止点为{target_pair['End_Node_Type']}，二者不一致，故跳过")
        return

    if not hdfs_file_exists(result_dir + "/_PathCoverInfo"):
        # 获得目标path的上的目标pair对应列名
        target_path_id_columns = [target_path_info[0]['head_node_std_name'], target_path_info[-1]['tail_node_std_name'], 'Feature_Time']
        
        # 获得目标pair上的目标pair对应的列名
        target_pair_id_columns = [target_pair['Start_Node_name'], target_pair['End_Node_name'], 'Feature_Time']
        
        # 为 DataFrame 设置别名，从而指定列的来源
        target_path_df = target_path_df.alias("target_path_df")
        target_pair_df = target_pair_df.alias("target_pair_df")
        
        # 基于这些列的join条件        
        join_conditions = [col(f"target_path_df.{target_path_id_columns[i]}") == col(f"target_pair_df.{target_pair_id_columns[i]}") for i in range(len(target_path_id_columns))]
        
        #################################################################################################################################
        # 以path为基准计算被cover的path
        path_cover_df = target_path_df.join(target_pair_df, join_conditions, "left")
        join_check_column = target_pair_id_columns[0]
        path_cover_df = path_cover_df.withColumn(f"cover_mark", col(f"target_pair_df.{join_check_column}").isNotNull())

        # 删除来自 target_pair_df 的所有列
        selected_columns = [col(f"target_path_df.{col_name}") for col_name in target_path_df.columns] + [col("cover_mark")]
        path_cover_df = path_cover_df.select(selected_columns)

        path_cover_df.persist(DEFAULT_STORAGE_LEVEL)
        #################################################################################################################################
        # 以pair为基准计算被cover的pair
        pair_cover_df = target_pair_df.join(target_path_df, join_conditions, "left")
        
        # 因为一个pair可能被同一种path中的多条cover,所以要计算的是该pair被该类path中的几条具体的path cover了
        join_check_column = target_path_id_columns[0]
        selected_columns = [col(f"target_pair_df.{col_name}") for col_name in target_pair_df.columns] + [col(f"target_path_df.{join_check_column}")]
        pair_cover_df = pair_cover_df.select(selected_columns)
        pair_cover_df = pair_cover_df.groupBy(target_pair_id_columns).agg(count(f"target_path_df.{join_check_column}").alias(f"path_{target_path_name}_cover_count"))
        
        pair_cover_df.persist(DEFAULT_STORAGE_LEVEL)
        #################################################################################################################################
        # 计算统计信息:总共有多少个path, cover了其中的多少，没cover的有多少,总共有多少个pair,cover了其中的多少，没cover的有多少
        path_cover_count = path_cover_df.filter(col("cover_mark") == True).count()
        path_uncover_count = path_cover_df.filter(col("cover_mark") == False).count()
        
        pair_cover_count = pair_cover_df.filter(col(f"path_{target_path_name}_cover_count") > 0).count()
        pair_uncover_count = pair_cover_df.filter(col(f"path_{target_path_name}_cover_count") == 0).count()
        
        cover_summary = {"path_cover_count": path_cover_count,
                    "path_uncover_count": path_uncover_count,
                    "path_count": (path_cover_count + path_uncover_count),
                    "path_success_rate": path_cover_count/(path_cover_count + path_uncover_count),
                    "pair_cover_count": pair_cover_count,
                    "pair_uncover_count": pair_uncover_count,
                    "pair_count": (pair_cover_count + pair_uncover_count),
                    "pair_success_rate": pair_cover_count/(pair_cover_count + pair_uncover_count)}
        
        print("cover summary:", cover_summary)
        #################################################################################################################################
        # 获得最优存储分区数
        
        # 保存结果
        path_cover_df.write.mode("overwrite").parquet(result_dir + "/Path_Cover")
        pair_cover_df.write.mode("overwrite").parquet(result_dir + "/Pair_Cover")

        # 释放persist的数据
        path_cover_df.unpersist()
        pair_cover_df.unpersist()
        
        # 保存path_cover和pair_cover的具体结果及统计信息
        cover_summary_str = json.dumps(cover_summary)
        hdfs_create_marker_file(result_dir, '_PathCoverInfo', cover_summary_str)
    else:
        # 直接读取结果
        cover_summary_str = hdfs_read_marker_file(result_dir, '_PathCoverInfo')
        cover_summary = json.loads(cover_summary_str)

        print("已完成该path的对比，cover summary:", cover_summary)
        
    return

def read_existing_paths(spark, new_paths_dir):
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
def Pyspark_K_Hop_Path_Sampling(spark, K_Hop_Target_Info, Subgraph_Config_dict, Subgraph_Hop_K, HDFS_Store_dir, target_pair = None):
    if hdfs_file_exists(HDFS_Store_dir + f"/{Subgraph_Hop_K-1}/_SUCCESS_Marker"):
        print("已完成目标节点全部k-hop路径的采样")
        return
    
    # 先获取目标起始节点及对应的信息
    tmp_target_node_df = K_Hop_Target_Info['Target_Node_df']
    tmp_target_node_type = K_Hop_Target_Info['Target_Node_Type']
    tmp_target_node_UID_name = K_Hop_Target_Info['Target_Node_UID_name']
    path_edge_times = K_Hop_Target_Info["Feature_Times"]
    
    print('将目标节点设为第0跳的seed_paths')
    
    # 将节点列名改成Node_0
    tmp_target_node_df = tmp_target_node_df.withColumnRenamed(tmp_target_node_UID_name, 'Node_0')
    
    tmp_target_node_info = {}
    tmp_target_node_info["tail_node_std_name"] = 'Node_0'
    tmp_target_node_info["tail_node_type"] = tmp_target_node_type
    seed_paths_list = [{"data": tmp_target_node_df, "path_config": [tmp_target_node_info]}]
    
    ##############################################################################################    
    # 依次处理各跳信息
    print("#####" * 20)
    for hop_k in range(Subgraph_Hop_K):
        print(f'开始处理第{hop_k}跳的关系')
        
        # 该跳对应路径信息的存储文件夹
        new_paths_dir = HDFS_Store_dir + f"/{hop_k}"
        
        # 获取该跳已存在的新路径的信息
        new_paths_list = read_existing_paths(spark, new_paths_dir)
                        
        ##############################################################################################
        # 检查是否已经完成对该跳的处理
        if hdfs_file_exists(new_paths_dir + "/_SUCCESS_Marker"):
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
            if hdfs_file_exists(new_paths_dir + f'/_SUCCESS_SEED_PATH_{seed_path_index}'):
                print(f'跳过,已完成第{seed_path_index}个seed_path能生成的全部路径的处理')
                continue

            print(f"开始处理第{seed_path_index}个seed_path:", seed_path["path_config"])

            # 遍历全部关系
            for add_edge_name in Subgraph_Config_dict:
                # 依次处理两种方向
                for edge_start_node in ["Head", "Tail"]:
                    # 获取该起始点对应的边的配置信息
                    add_edge_config = get_add_edge_config(Subgraph_Config_dict, add_edge_name, edge_start_node, hop_k)

                    print(f'尝试从{edge_start_node}方向为该path添加上边{add_edge_name}:', add_edge_config)
                    
                    # 检查该配置是否需要进行采样（点的类型是否符合要求，以及配置里是否要求进行采样）
                    if seed_path['path_config'][-1]['tail_node_type'] != add_edge_config["head_node_type"]:
                        print('edge and seed path not match')
                        continue

                    if "Edge_Neighbor_Limits" in add_edge_config and add_edge_config["Edge_Neighbor_Limits"]["Max_Num"] <= 0:
                        print('Skip this edge, since neighbor number limits to 0')
                        continue

                    # 若当前已是最后一跳，且终点不为目标pair的终点类型，则跳过
                    if hop_k == (Subgraph_Hop_K - 1) and target_pair is not None:
                        if target_pair['End_Node_Type'] != add_edge_config['tail_node_type']:
                            print('最后一跳的终止节点不符合，故跳过')
                            continue

                    # 开始生成new_path
                    new_path = generate_new_path(spark, hop_k, seed_path_index, seed_path, add_edge_name, add_edge_config, 
                                        path_edge_times, new_paths_dir, len(new_paths_list), target_pair)

                    # 记录结果
                    new_paths_list.append(new_path)

            hdfs_create_marker_file(new_paths_dir, f'_SUCCESS_SEED_PATH_{seed_path_index}')
            print(f'完成第{seed_path_index}个seed_path能生成的全部路径的处理')

        hdfs_create_marker_file(new_paths_dir)
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

def k_hop_path_pairing_to_sample(spark, paths_result_dir, max_hop_k, target_pair, max_main_path = 5):
    """
    获得完整的全部pair和全部path之间的匹配情况，返回主要路径和辅助路径的采样方式，以及正负样本的信息
    
    输入：
        数据的存储路径
        要检测的文件的跳数
    
    返回：
        正负样本
        与样本有关联的对应的路径信息
        主路径对应的配置信息
        辅助路径对应的配置信息
    """
    # 如果已完成采样则直接返回
    if hdfs_file_exists(paths_result_dir + "/Sample/_SUCCESS_Marker"):
        all_sample_pairs = spark.read.parquet(paths_result_dir + "/Sample/Sample_Pairs")
        
        main_paths = []
        main_path_result_dirs = hdfs_list_files(paths_result_dir + "/Sample/Main_Paths")
        for main_path_result_dir in main_path_result_dirs:
            main_path = {}
            
            main_path['path_name'] = main_path_result_dir.split('/')[-1]
            
            print("Read main path:", main_path['path_name'])
            
            # 读取数据
            main_path['data'] = spark.read.parquet(main_path_result_dir + "/Path_Data")
            main_path['Path_Cover'] = spark.read.parquet(main_path_result_dir + "/Path_Cover")
            main_path['Pair_Cover'] = spark.read.parquet(main_path_result_dir + "/Pair_Cover")

            path_config_str = hdfs_read_marker_file(main_path_result_dir, '_PATH_CONFIG')
            main_path['path_config'] = json.loads(path_config_str)
            
            cover_summary_str = hdfs_read_marker_file(main_path_result_dir, '_PathCoverInfo')
            main_path['Cover_Summary'] = json.loads(cover_summary_str)
            
            main_paths.append(main_path)
        
        filtered_aux_paths = []
        aux_path_result_dirs = hdfs_list_files(paths_result_dir + "/Sample/Auxiliary_Paths")
        for aux_path_result_dir in aux_path_result_dirs:
            aux_path = {}
            
            aux_path['path_name'] = aux_path_result_dir.split('/')[-1]
            
            print("Read auxiliary path:", aux_path['path_name'])
            
            # 读取数据
            aux_path['data'] = spark.read.parquet(aux_path_result_dir + "/Path_Data")
            aux_path['Path_Cover'] = spark.read.parquet(aux_path_result_dir + "/Path_Cover")
            aux_path['Pair_Cover'] = spark.read.parquet(aux_path_result_dir + "/Pair_Cover")

            path_config_str = hdfs_read_marker_file(aux_path_result_dir, '_PATH_CONFIG')
            aux_path['path_config'] = json.loads(path_config_str)
            
            cover_summary_str = hdfs_read_marker_file(aux_path_result_dir, '_PathCoverInfo')
            aux_path['Cover_Summary'] = json.loads(cover_summary_str)
            
            filtered_aux_paths.append(aux_path)
        
        return_data = {}
        return_data['sample_pairs'] = all_sample_pairs
        return_data['main_paths'] = main_paths
        return_data['aux_paths'] = filtered_aux_paths
        
        return return_data
    ##############################################################################################
    
    valid_paths_list = []
    
    for hop_k in range(max_hop_k):
        # 该跳对应路径信息的存储文件夹
        current_hop_paths_dir = paths_result_dir + f"/{hop_k}"
        
        # 获取该跳下全部的path对应的子文件夹
        current_hop_paths_dirs_list = hdfs_list_files(current_hop_paths_dir)
        current_hop_paths_dirs_list = [tmp_path_dir for tmp_path_dir in current_hop_paths_dirs_list if tmp_path_dir.split('/')[-1].isdigit()]
        
        # 依次处理各个path对应的数据
        for path_dir in current_hop_paths_dirs_list:
            # 只处理有对应的覆盖信息的路径的相关信息
            if hdfs_file_exists(path_dir + "/_PathCoverInfo"):
                path_index = int(path_dir.split('/')[-1])
                path_name = f'{hop_k}_{path_index}'

                # 读取对应的path数据
                path_df = spark.read.parquet(path_dir + '/Path_Data')
                
                # 读取对应的path_info
                path_config_str = hdfs_read_marker_file(path_dir, '_PATH_CONFIG')
                path_config = json.loads(path_config_str)
                
                # 读取对应的path_cover和pair_cover
                path_cover_df = spark.read.parquet(path_dir + '/Path_Cover')
                pair_cover_df = spark.read.parquet(path_dir + '/Pair_Cover')

                # 读取对应的覆盖情况
                cover_summary_str = hdfs_read_marker_file(path_dir, '_PathCoverInfo')
                cover_summary = json.loads(cover_summary_str)
                
                # 保留结果: name, info, path_cover, pair_cover, cover_summary
                valid_path = {}
                valid_path['path_name'] = path_name
                valid_path['data'] = path_df
                valid_path['path_config'] = path_config
                valid_path['Path_Cover'] = path_cover_df
                valid_path['Pair_Cover'] = pair_cover_df
                valid_path['Cover_Summary'] = cover_summary
                
                # 打印该路径的覆盖结果
                print(f"路径{path_name}的覆盖情况为{cover_summary}，该路径对应的具体组成为{path_config}")
                
                if valid_path['Cover_Summary']['path_cover_count'] > 10:
                    valid_paths_list.append(valid_path)
                else:
                    print('该路径不合格，跳过')
    ##############################################################################################
    if not hdfs_file_exists(path_dir + "/Pair_Cover_Summary/_SUCCESS"):
        # 获取各个pair的id列
        target_pair_id_columns = [target_pair['Start_Node_name'], target_pair['End_Node_name'], 'Feature_Time']

        # 合并pair_cover
        pair_cover_all_df = valid_paths_list[0]['Pair_Cover']
        for valid_path in valid_paths_list[1:]:
            pair_cover_all_df = pair_cover_all_df.join(valid_path['Pair_Cover'], target_pair_id_columns, 'left')

        # 标记各个pair被各种路径中的几条path join上了，以及被几种path join上了
        valid_paths_names = [valid_path['path_name'] for valid_path in valid_paths_list]
        all_path_cover_columns = [f"path_{path_name}_cover_count" for path_name in valid_paths_names]
        all_paths_kinds_count = " + ".join([f"(CASE WHEN {c} > 0 THEN 1 ELSE 0 END)" for c in all_path_cover_columns])
        all_paths_count = " + ".join(all_path_cover_columns)
        pair_cover_all_df = pair_cover_all_df.withColumn("all_paths_kinds_count", expr(all_paths_kinds_count))\
                                 .withColumn("all_paths_count", expr(all_paths_count))
        pair_cover_all_df.persist(DEFAULT_STORAGE_LEVEL)

        pair_cover_all_df.write.mode("overwrite").parquet(paths_result_dir + '/Pair_Cover_Summary')
    else:
        pair_cover_all_df = spark.read.parquet(paths_result_dir + '/Pair_Cover_Summary')
    
    ##############################################################################################
    # 获得全部被覆盖的pair的信息
    covered_pair_df = pair_cover_all_df.filter(pair_cover_all_df.all_paths_kinds_count > 0)
    
    # 输出总共覆盖了多少个pair
    all_covered_pair_count = covered_pair_df.count()
    print('总共覆盖的pair数:', all_covered_pair_count)
    
    # 获得目标起点对应的列名
    start_node_id_columns = [target_pair['Start_Node_name'], 'Feature_Time']
    
    # 获得被覆盖的起点
    covered_pair_start_df = covered_pair_df.select(start_node_id_columns).distinct()
    covered_pair_start_df.persist(DEFAULT_STORAGE_LEVEL)
    
    covered_pair_start_df = covered_pair_start_df.alias("covered_pair_start_df")
    
    print("#####" * 20)

    for valid_path in valid_paths_list:
        # 获得目标path的上的目标pair对应列名
        path_start_node_columns = [valid_path['path_config'][0]['head_node_std_name'], 'Feature_Time']
        
        # 为 DataFrame 设置别名，从而指定列的来源
        valid_path_df = valid_path['data'].alias("valid_path_df")
        
        # 基于这些列的join条件        
        join_conditions = [col(f"valid_path_df.{path_start_node_columns[i]}") == col(f"covered_pair_start_df.{start_node_id_columns[i]}") for i in range(len(path_start_node_columns))]
        
        # 只保留这些pair对应的公司为起点的路径
        valid_path['data'] = valid_path_df.join(covered_pair_start_df, on = join_conditions, how = 'inner')
        
        # 只保留valid_path_df里的列
        selected_columns = [col(f"valid_path_df.{col_name}") for col_name in valid_path_df.columns]
        valid_path['data'] = valid_path['data'].select(selected_columns)
        
        valid_path['data'].persist(DEFAULT_STORAGE_LEVEL)
        
        # 获取过滤后的总数
        print(valid_path['path_name'], valid_path['data'].count())
        
        # 更新path_uncover_count path_success_rate
        
        
    ##############################################################################################
    # 按各路径的匹配成功率进行排序
    sorted_valid_paths_list = sorted(valid_paths_list, key = lambda x: x['Cover_Summary']['path_success_rate'], reverse = True)
    
    # 获取主路径和辅助路径
    main_paths = []
    aux_paths = []
    for valid_path in sorted_valid_paths_list:
        # 将匹配路径数大于200且成功率大于0.0001的前5条路径的作为主路径
        if len(main_paths) < max_main_path and valid_path['Cover_Summary']['path_cover_count'] > 50 and valid_path['Cover_Summary']['path_success_rate'] > 0.0001:
            main_paths.append(valid_path)
        else:
            # 将剩余的匹配路径数大于10的路径作为辅助路径
            aux_paths.append(valid_path)
            
    print("Main paths name and cover:")
    for path in main_paths:
        print(path['path_name'], path['Cover_Summary'])
        
    print("-----" * 20)
    
    print("Auxiliary paths name and cover:")
    for path in aux_paths:
        print(path['path_name'], path['Cover_Summary'])
    
    print("#####" * 20)
    ##############################################################################################
    # 标记各个pair是否被主路径join上了
    main_paths_names = [path['path_name'] for path in main_paths]
    main_path_cover_columns = [f"path_{path_name}_cover_count" for path_name in main_paths_names]
    main_paths_kinds_count = " + ".join([f"(CASE WHEN {c} > 0 THEN 1 ELSE 0 END)" for c in main_path_cover_columns])
    main_paths_count = " + ".join(main_path_cover_columns)
    pair_cover_all_df = pair_cover_all_df.withColumn("main_paths_kinds_count", expr(main_paths_kinds_count))\
                             .withColumn("main_paths_count", expr(main_paths_count))
    
    # 将被join上的pair作为正样本
    positive_samples = pair_cover_all_df.filter(col("main_paths_count") > 0)
    positive_samples = positive_samples.select(target_pair_id_columns)
    positive_samples = positive_samples.withColumn("Label", lit(1))
    
    # 获得主路径涉及的全部pair
    sample_pairs_list = []
    for main_path in main_paths:
        # 获得pair在该path中对应的全部id列对应的列名
        path_id_columns = [main_path['path_config'][0]['head_node_std_name'], main_path['path_config'][-1]['tail_node_std_name'], 'Feature_Time']
        
        # 只保留id列
        main_path_pairs = main_path['data'].select(path_id_columns)
        
        # 修正id列列名
        main_path_pairs = main_path_pairs.withColumnRenamed(path_id_columns[0], target_pair_id_columns[0])
        main_path_pairs = main_path_pairs.withColumnRenamed(path_id_columns[1], target_pair_id_columns[1])
        
        # 去重
        main_path_pairs = main_path_pairs.distinct()
        
        sample_pairs_list.append(main_path_pairs)
        
    # 合并结果,并去重
    all_sample_pairs = reduce(DataFrame.union, sample_pairs_list).distinct()
    
    # 标记出正样本,并将剩下的作为负样本
    all_sample_pairs = all_sample_pairs.join(positive_samples, target_pair_id_columns, 'left')
    all_sample_pairs = all_sample_pairs.fillna({'Label': 0})
    all_sample_pairs.persist(DEFAULT_STORAGE_LEVEL)
    
    # 输出正负样本个数
    positive_sample_count = all_sample_pairs.filter(col("Label") == 1).count()
    negative_sample_count = all_sample_pairs.filter(col("Label") == 0).count()
    print(f"正样本个数为{positive_sample_count},负样本个数为{negative_sample_count}")
    
    all_sample_pairs.write.mode("overwrite").parquet(paths_result_dir + "/Sample/Sample_Pairs")
    ##############################################################################################
    # 为样本data设置别名
    all_sample_pairs = all_sample_pairs.alias("all_sample_pairs")
    
    # 保留辅助路径中和全量样本中匹配的路径作为补充信息
    filtered_aux_paths = []
    for aux_path in aux_paths:
        # 获得pair在该path中对应的全部id列对应的列名
        path_id_columns = [aux_path['path_config'][0]['head_node_std_name'], aux_path['path_config'][-1]['tail_node_std_name'], 'Feature_Time']
        
        # 为 DataFrame 设置别名，从而指定列的来源
        aux_path_df = aux_path['data'].alias("aux_path_df")
        
        # 记录该path本身的列
        selected_columns = [col(f"aux_path_df.{col_name}") for col_name in aux_path_df.columns]
        
        # 基于这些列的join条件        
        join_conditions = [col(f"aux_path_df.{path_id_columns[i]}") == col(f"all_sample_pairs.{target_pair_id_columns[i]}") for i in range(len(path_id_columns))]
        
        # 只保留样本对应的行
        aux_path_df = aux_path_df.join(all_sample_pairs, join_conditions, "inner")
        
        # 只保留path本身的列
        aux_path_df = aux_path_df.select(selected_columns)
        
        # 存储对应信息
        filtered_aux_path = aux_path.copy()
        filtered_aux_path['data'] = aux_path_df
        
        filtered_aux_paths.append(filtered_aux_path)
    
    ##############################################################################################
    # 保存样本，主路径和辅助路径的信息，以及辅助路径中保留的路径
    # all_sample_pairs, main_paths, filtered_aux_paths
    
    for main_path in main_paths:
        path_name = main_path['path_name']
        main_path_result_dir = paths_result_dir + f"/Sample/Main_Paths/{path_name}"
        
        # 保存数据
        main_path['data'].write.mode("overwrite").parquet(main_path_result_dir + "/Path_Data")
        main_path['Path_Cover'].write.mode("overwrite").parquet(main_path_result_dir + "/Path_Cover")
        main_path['Pair_Cover'].write.mode("overwrite").parquet(main_path_result_dir + "/Pair_Cover")

        path_config_str = json.dumps(main_path['path_config'])
        hdfs_create_marker_file(main_path_result_dir, '_PATH_CONFIG', path_config_str)
        
        cover_summary_str = json.dumps(main_path['Cover_Summary'])
        hdfs_create_marker_file(main_path_result_dir, '_PathCoverInfo', cover_summary_str)
        
    for aux_path in filtered_aux_paths:
        path_name = aux_path['path_name']
        aux_path_result_dir = paths_result_dir + f"/Sample/Auxiliary_Paths/{path_name}"
        
        # 保存数据
        aux_path['data'].write.mode("overwrite").parquet(aux_path_result_dir + "/Path_Data")
        aux_path['Path_Cover'].write.mode("overwrite").parquet(aux_path_result_dir + "/Path_Cover")
        aux_path['Pair_Cover'].write.mode("overwrite").parquet(aux_path_result_dir + "/Pair_Cover")
       
        path_config_str = json.dumps(aux_path['path_config'])
        hdfs_create_marker_file(aux_path_result_dir, '_PATH_CONFIG', path_config_str)
        
        cover_summary_str = json.dumps(aux_path['Cover_Summary'])
        hdfs_create_marker_file(aux_path_result_dir, '_PathCoverInfo', cover_summary_str)
    
    # 创建完成标志
    hdfs_create_marker_file(paths_result_dir + '/Sample')
    
    return_data = {}
    return_data['sample_pairs'] = all_sample_pairs
    return_data['main_paths'] = main_paths
    return_data['aux_paths'] = filtered_aux_paths

    return return_data