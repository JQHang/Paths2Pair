import sys
import os
import re
import gc
import io
import time
import copy
import json
import math
import random

os.environ['SPARK_HOME']="/software/servers/10k/mart_scr/spark_3.0"
os.environ['PYTHONPATH']="/software/servers/10k/mart_scr/spark_3.0/python:/software/servers/10k/mart_scr/spark_3.0/python/lib/py4j-0.10.9-src.zip"
os.environ['LD_LIBRARY_PATH']="/software/servers/jdk1.8.0_121/lib:/software/servers/jdk1.8.0_121/jre/lib/amd64/server:/software/servers/hope/mart_sch/hadoop/lib/native"
os.environ['PYSPARK_PYTHON']="/usr/local/anaconda3/bin/python3.6"
os.environ['PYSPARK_DRIVER_PYTHON']="/usr/local/anaconda3/bin/python3.6"

sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python/lib/py4j-0.10.9-src.zip')
sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python')
sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python/lib/pyspark.zip')

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.types import IntegerType, FloatType, DoubleType, DecimalType, LongType, ShortType, ByteType
from pyspark import StorageLevel, SparkContext
from pyspark.sql import Row
from pyspark.sql.functions import when, lit, udf, array
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, lit, broadcast
from pyspark.sql.types import StringType
from pyspark.sql import HiveContext,SparkSession

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import SparseVector, DenseVector, VectorUDT, Vectors

import json
import random
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import pyarrow as pa
from operator import add
from pyarrow import fs as pafs
from py4j.java_gateway import java_import
from py4j.protocol import Py4JError, Py4JJavaError, Py4JNetworkError

DEFAULT_STORAGE_LEVEL = StorageLevel.MEMORY_AND_DISK
# DEFAULT_STORAGE_LEVEL = StorageLevel.DISK_ONLY

# Data_In_HDFS_Path = '/user/mart_coo/mart_coo_map/hangjinquan/Complex_Path'
Data_In_HDFS_Path = '/user/mart_coo/mart_coo_innov/Complex_Path'

"""
作用：
    启动pyspark

输入：
    version：pyspark版本
    
返回值：
    Spark_Session
"""
def Start_Spark(config_dict={}):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print('Start Create Spark Session', current_time)
    
    # 初始化 SparkSession 的构建器
    builder = SparkSession.builder \
              .appName(f"SparkSession at {current_time}") \
              .enableHiveSupport()

    # 默认配置
    default_configs = {
        "spark.default.parallelism": "600",
        "spark.sql.shuffle.partitions": "1200",
        "spark.sql.broadcastTimeout": "3600",
        "spark.driver.memory": "40g",
        "spark.executor.memory": "40g",
        "spark.executor.cores": "4",
        "spark.executor.instances": "150",
        "spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class": "DockerLinuxContainer",
        "spark.executorEnv.yarn.nodemanager.container-executor.class": "DockerLinuxContainer",
        "spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name": "bdp-docker.jd.com:5000/wise_mart_bag:latest",
        "spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name": "bdp-docker.jd.com:5000/wise_mart_bag:latest",
        "spark.sql.crossJoin.enabled": "true",
        "spark.driver.maxResultSize": "0",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.driver.allowMultipleContexts": "false",
        "spark.sql.autoBroadcastJoinThreshold": str(100 * 1024 * 1024),
        "spark.sql.adaptive.shuffle.targetPostShuffleInputSize": str(300 * 1024 * 1024)
    }

    # 更新默认配置为用户提供的配置
    for key, value in config_dict.items():
        default_configs[key] = value

    # 将更新后的配置应用到 SparkSession 的构建器
    for key, value in default_configs.items():
        builder = builder.config(key, value)

    # 创建 SparkSession
    Spark_Session = builder.getOrCreate()

    print('Spark Session Created', datetime.now())
    
    # 获取 Spark 应用程序的 ID
    tmp_app_id = Spark_Session.sparkContext.applicationId

    # 打印应用程序 ID
    print("Application ID: http://10k2.jd.com/proxy/" + tmp_app_id)

    # 历史记录查询位置
    print(f"History Record: http://10k.sparkhs-3.jd.com/history/{tmp_app_id}/stages/")

    # 获取 Hadoop Configuration
    hadoop_conf = Spark_Session._jsc.hadoopConfiguration()

    # 获取 ResourceManager 地址
    rm_address = hadoop_conf.get("yarn.resourcemanager.address")
    print("ResourceManager Address:", rm_address)
    
    return Spark_Session


class ResilientSparkRunner:
    
    def __init__(self, config_dict=None, max_restarts=20):
        self.spark_restart_count = 0
        self.max_restarts = max_restarts
        self.config_dict = config_dict or {}
        self.Spark_Session = self._start_spark()
    
    def _start_spark(self):
        return Start_Spark(self.config_dict)

    def _handle_error(self, e):
        print('*******************************************************************************')
        print("An error occurred:", str(e))

        # 检查Spark会话是否处于活跃状态
        if self.Spark_Session and self.Spark_Session.sparkContext:
            try:
                self.Spark_Session.stop()
            except Exception as stop_exception:
                print("Error stopping SparkSession:", str(stop_exception))
        
        self.Spark_Session._instantiatedContext = None
        self.Spark_Session = None
        gc.collect()

        interrupt_time = datetime.now()
        print('pyspark异常中断，时间:', interrupt_time)

        if interrupt_time.hour < 8 or (interrupt_time.hour == 8 and interrupt_time.minute < 59) or (interrupt_time.hour == 23 and interrupt_time.minute > 55):
            tmp_interrupt_time_hour = interrupt_time.hour
            tmp_interrupt_time_minute = interrupt_time.minute
            
            if tmp_interrupt_time_hour == 23:
                tmp_interrupt_time_hour = -1
            
            print('0-9点无法运算，故开始等待')
            time_sleep = (8 - tmp_interrupt_time_hour) * 3600 + (59 - tmp_interrupt_time_minute) * 60
            print('休眠时间', time_sleep)
            
            time.sleep(time_sleep)
            print('已到早上9点，重启运算')
        elif self.spark_restart_count > self.max_restarts:
            print(f'重启超过{self.max_restarts}次，故终止')
            return False
        else:
            self.spark_restart_count += 1
            print('重启Spark并重新开始运算')
        
        time.sleep(5)
        
        # 重启Spark会话
        try:
            self.Spark_Session = self._start_spark()
        except Exception as start_exception:
            print("Error starting SparkSession:", str(start_exception))
            return False

        print('*******************************************************************************')
        return True

    def run(self, task_function, *args, **kwargs):
        result = None
        while True:
            try:
                # 传递启动的spark和参数和关键字参数给task_function
                result = task_function(self.Spark_Session, *args, **kwargs)
                break
            except (Py4JError, Py4JJavaError, Py4JNetworkError) as e:
                should_continue = self._handle_error(e)
                if not should_continue:
                    break
        return result
    
def path_exists_on_hdfs(spark: SparkSession, hdfs_path: str) -> bool:
    """
    Check if a path exists on HDFS.

    Parameters:
    - spark: An active SparkSession.
    - hdfs_path: The HDFS path to check.

    Returns:
    - True if the path exists on HDFS, False otherwise.
    """

    # 导入需要的 Java 类
    java_import(spark._jvm, 'org.apache.hadoop.fs.Path')
    java_import(spark._jvm, 'org.apache.hadoop.fs.FileSystem')
    
    # 获取 HDFS 配置
    hadoop_conf = spark._jsc.hadoopConfiguration()

    # 获取文件系统对象
    fs = spark._jvm.FileSystem.get(hadoop_conf)

    # 要检查的路径
    path = spark._jvm.Path(hdfs_path)

    # 检查路径是否存在
    return fs.exists(path)

def list_files_on_hdfs(spark: SparkSession, hdfs_path: str) -> list:
    """
    List all files in a directory on HDFS.

    Parameters:
    - spark: An active SparkSession.
    - hdfs_path: The HDFS directory path to list files.

    Returns:
    - A list of file paths in the specified HDFS directory.
    """
    
    # 导入需要的 Java 类
    java_import(spark._jvm, 'org.apache.hadoop.fs.Path')
    java_import(spark._jvm, 'org.apache.hadoop.fs.FileSystem')
    java_import(spark._jvm, 'org.apache.hadoop.fs.FileStatus')
    
    # 获取 HDFS 配置
    hadoop_conf = spark._jsc.hadoopConfiguration()

    # 获取文件系统对象
    fs = spark._jvm.FileSystem.get(hadoop_conf)

    # 要列出的目录路径
    path = spark._jvm.Path(hdfs_path)

    # 列出路径下的文件
    file_statuses = fs.listStatus(path)
    file_paths = [file_status.getPath().toString() for file_status in file_statuses]

    return file_paths

"""
作用：
    根据指定表名和格式创建表(若已存在该表，则会进行drop)

输入：
    Spark_Session：pyspark接口
    Table_Name：目标表名
    Table_Columns_List：目标表列名
    Table_Columns_Type_List: 目标表列的类型
    Table_Columns_Comment_List: 目标表列的注释
    
返回值：
    无
"""
def Pyspark_Create_Table(Spark_Session, Table_Name, Table_Columns_List, Table_Columns_Type_List, Table_Columns_Comment_List = None):
    tmp_sql = """CREATE TABLE IF NOT EXISTS """ + Table_Name + """ ( """
    
    for tmp_column_i in range(len(Table_Columns_List)):
        tmp_column = Table_Columns_List[tmp_column_i]
        tmp_column_type = Table_Columns_Type_List[tmp_column_i]
        
        tmp_sql = tmp_sql + tmp_column + ' ' + tmp_column_type
        
        if Table_Columns_Comment_List != None:
            tmp_column_comment = Table_Columns_Comment_List[tmp_column_i]
            tmp_sql = tmp_sql + ' COMMENT \"' + tmp_column_comment + "\""
        
        if tmp_column_i != (len(Table_Columns_List) - 1):
            tmp_sql = tmp_sql + ', '
                
    tmp_sql = tmp_sql + """ )
                PARTITIONED BY
                (
                    dt string
                )
                stored AS orc tblproperties
                 (
                     'orc.compress' = 'SNAPPY'
                 )"""
    
    Spark_Session.sql(tmp_sql)
    
    return

"""
作用：
    根据指定表名和数据创建表并上传

输入：
    Spark_Session：pyspark接口
    Table_Name：目标表名
    Data_To_Upload:要上传的数据
    Upload_table_dt:目标时间分区
    Table_Columns_List：目标表列名
    Table_Columns_Comment_List: 目标表列的注释
    
返回值：
    无
"""
def Upload_RDD_Data_to_Database(Spark_Session, Table_Name, Data_To_Upload, Upload_table_dt, Set_Table_Columns_List = [], 
                      Set_Table_Columns_Comment_List = [], batch_count = 1):
    upload_start_time = datetime.now()
    
    # Get Table Info
    Table_Columns_List = []
    Table_Columns_Type_List = []
    tmp_upload_table_rdd_json = json.loads(Data_To_Upload.schema.json())['fields']
    for tmp_col_info in tmp_upload_table_rdd_json:
        col = tmp_col_info['name']
        col_type = tmp_col_info['type']

        Table_Columns_List.append(col)
        Table_Columns_Type_List.append(col_type)
    
    # Change column name if len is equal
    if len(Table_Columns_List) == len(Set_Table_Columns_List):
        Table_Columns_List = Set_Table_Columns_List
    
    # Clear comment list if len isnot equal
    if len(Table_Columns_List) == len(Set_Table_Columns_Comment_List):
        # Create Table
        Pyspark_Create_Table(Spark_Session, Table_Name, Table_Columns_List, Table_Columns_Type_List, Set_Table_Columns_Comment_List)
    else:
        Pyspark_Create_Table(Spark_Session, Table_Name, Table_Columns_List, Table_Columns_Type_List)

    if batch_count > 1:
        # 定义批次大小的比例
        batch_ratios = [1.0 / batch_count] * (batch_count - 1)
        batch_ratios.append(1.0 - sum(batch_ratios))

        # 使用 randomSplit 将 DataFrame 分成多个小 DataFrame
        Data_To_Upload_list = Data_To_Upload.randomSplit(batch_ratios)
    else:
        Data_To_Upload_list = [Data_To_Upload]
        
    for idx, small_df in tqdm(enumerate(Data_To_Upload_list), total=len(Data_To_Upload_list), desc="Uploading batches"):
        # 为每个小 DataFrame 创建一个临时视图
        view_name = f"temp_view_{idx}"
        small_df.createOrReplaceTempView(view_name)

        # 对于第一个 DataFrame, 使用 INSERT OVERWRITE。对于后续的 DataFrame, 使用 INSERT INTO。
        if idx == 0:
            query = f"""
                INSERT OVERWRITE TABLE {Table_Name} PARTITION (dt='{Upload_table_dt}')
                SELECT * FROM {view_name}
            """
        else:
            query = f"""
                INSERT INTO TABLE {Table_Name} PARTITION (dt='{Upload_table_dt}')
                SELECT * FROM {view_name}
            """
            
        Spark_Session.sql(query)
    
    Spark_Session.catalog.dropTempView(view_name)
    
    upload_end_time = datetime.now()
    print('完成目标表的上传, 上传函数消耗时间为:', (upload_end_time - upload_start_time))
    
    return

"""
作用：
    针对指定的rdd数据，计算groupby后的结果

输入：
    Spark_Session：pyspark接口
    Aim_Table_Rdd：目标pyspark数据
    Groupby_Column_List: 目标列列名
    Feature_Columns_List：目标特征列
    Groupby_Type_List：目标groupby运算方式

返回值：
    采样后的pyspark数据
"""
def Groupby_Feature_Table(Spark_Session, Aim_Table_Rdd, Groupby_Column_List, Feature_Columns_List, Groupby_Type_List):
    # 保证聚合数据没有0值
    Aim_Table_Rdd = Aim_Table_Rdd.fillna(0)
    
    Aim_Table_Rdd.createOrReplaceTempView("EMP")

    agg_functions = []
    for tmp_groupby_type in Groupby_Type_List:
        if tmp_groupby_type in ['AVG', 'SUM', 'MAX', 'MIN']:
            for tmp_feature_column in Feature_Columns_List:
                agg_function = f"{tmp_groupby_type}({tmp_feature_column}) AS {tmp_groupby_type}_{tmp_feature_column}"
                agg_functions.append(agg_function)
        elif tmp_groupby_type == 'COUNT':
            agg_functions.append("COUNT(*) AS Groupby_COUNT")

    # 构建 SQL 查询字符串
    if isinstance(Groupby_Column_List, list):
        sql_str = "SELECT " + ",".join(Groupby_Column_List) + ", " + ", ".join(agg_functions) + f" FROM EMP GROUP BY " + ",".join(Groupby_Column_List)
    elif isinstance(Groupby_Column_List, str):
        sql_str = "SELECT " + Groupby_Column_List + ", " + ", ".join(agg_functions) + f" FROM EMP GROUP BY {Groupby_Column_List}"
        
    return Spark_Session.sql(sql_str)

def pyspark_feature_aggregation(spark, df, id_columns, feat_columns, agg_functions):
    # 保证聚合数据没有0值
    df = df.fillna(0)
    
    df.createOrReplaceTempView("EMP")

    sql_agg_functions = []
    feature_columns = []
    feature_comments = []
    for agg_function in agg_functions:
        if agg_function in ['AVG', 'SUM', 'MAX', 'MIN']:
            for feat_index, feat_column in enumerate(feat_columns):
                sql_agg_function = f"{agg_function}({feat_column}) AS {agg_function}_{feat_column}"
                sql_agg_functions.append(sql_agg_function)
                feature_columns.append(f"{agg_function}_{feat_column}")
                feature_comments.append(f"({agg_function}){feat_column}")
        elif tmp_groupby_type == 'COUNT':
            sql_agg_functions.append("COUNT(*) AS COUNT_All")
            feature_columns.append(f"COUNT_All")
            feature_comments.append(f"COUNT_All")
                
    # 构建 SQL 查询字符串
    sql_str = "SELECT " + ",".join(id_columns) + ", " + ", ".join(sql_agg_functions) + f" FROM EMP GROUP BY " + ",".join(id_columns)
    
    aggregated_df = spark.sql(sql_str)
    
    aggregated_result = {}
    aggregated_result['data'] = aggregated_df
    aggregated_result['feature_columns'] = feature_columns
    aggregated_result['feature_comments'] = feature_comments
    
    return aggregated_result

# def Groupby_Feature_Table(Spark_Session, Aim_Table_Rdd, Aim_Column_name, Feature_Columns_List, Groupby_Type_List):
#     # 转换为键值对 RDD
#     pair_rdd = Aim_Table_Rdd.rdd.map(lambda row: (row[Aim_Column_name], [row[col] for col in Feature_Columns_List]))

#     # 定义聚合函数
#     def aggregate_values(a, b):
#         result = []
#         for i in range(len(a)):
#             if 'SUM' in Groupby_Type_List or 'AVG' in Groupby_Type_List:
#                 result.append(a[i] + b[i])  # SUM 或 AVG
#             elif 'MAX' in Groupby_Type_List:
#                 result.append(max(a[i], b[i]))  # MAX
#             elif 'MIN' in Groupby_Type_List:
#                 result.append(min(a[i], b[i]))  # MIN
#         return result

#     # 使用 reduceByKey 进行聚合
#     aggregated_rdd = pair_rdd.reduceByKey(aggregate_values)
    
#     # 如果需要 AVG，需要计算总数
#     if 'AVG' in Groupby_Type_List:
#         count_rdd = pair_rdd.mapValues(lambda x: 1).reduceByKey(add)
#         aggregated_rdd = aggregated_rdd.join(count_rdd).mapValues(lambda x: [val / x[1] for val in x[0]])
        
#     # 转换回 DataFrame
#     def to_row(key, values):
#         row_dict = {Aim_Column_name: key}
#         for i, col in enumerate(Feature_Columns_List):
#             for agg_type in Groupby_Type_List:
#                 if agg_type == 'COUNT':
#                     row_dict[agg_type + '_' + col] = values[i]
#                 else:
#                     row_dict[agg_type + '_' + col] = values[i]
#         return Row(**row_dict)

#     result_rdd = aggregated_rdd.map(lambda x: to_row(x[0], x[1]))
#     result_df = Spark_Session.createDataFrame(result_rdd)

#     return result_df

# def Groupby_Feature_Table(Spark_Session, Aim_Table_Rdd, Aim_Column_name, Feature_Columns_List, Groupby_Type_List):
#     # 将 DataFrame 转换为键值对 RDD
#     pair_rdd = Aim_Table_Rdd.rdd.map(lambda row: (row[Aim_Column_name], [row[col] for col in Feature_Columns_List]))

#     # 使用 groupByKey 进行分组
#     grouped_rdd = pair_rdd.groupByKey()

#     # 定义聚合函数
#     def aggregate(group):
#         # 初始化聚合结果
#         sums = [0] * len(Feature_Columns_List)
#         mins = [float('inf')] * len(Feature_Columns_List)
#         maxs = [float('-inf')] * len(Feature_Columns_List)
#         count = 0

#         for values in group:
#             count += 1
#             for i, value in enumerate(values):
#                 sums[i] += value
#                 mins[i] = min(mins[i], value)
#                 maxs[i] = max(maxs[i], value)

#         avgs = [sums[i] / count for i in range(len(sums)) if count > 0]
#         return (sums, mins, maxs, avgs, count)

#     # 对每个组应用聚合函数
#     result_rdd = grouped_rdd.mapValues(aggregate)

#     # 将结果转换回 DataFrame
#     def to_row(key, values):
#         sums, mins, maxs, avgs, count = values
#         row_dict = {Aim_Column_name: key, "count": count}
#         for i, col in enumerate(Feature_Columns_List):
#             if 'SUM' in Groupby_Type_List:
#                 row_dict[f"sum_{col}"] = sums[i]
#             if 'MIN' in Groupby_Type_List:
#                 row_dict[f"min_{col}"] = mins[i]
#             if 'MAX' in Groupby_Type_List:
#                 row_dict[f"max_{col}"] = maxs[i]
#             if 'AVG' in Groupby_Type_List:
#                 row_dict[f"avg_{col}"] = avgs[i]
#         return row_dict

#     result_df = Spark_Session.createDataFrame(result_rdd.map(lambda x: to_row(x[0], x[1])))

#     return result_df


# def Groupby_Feature_Table(Spark_Session, Aim_Table_Rdd, Aim_Column_name, Feature_Columns_List, Groupby_Type_List):
#     # 转换为键值对 RDD
#     pair_rdd = Aim_Table_Rdd.rdd.map(lambda row: (row[Aim_Column_name], [row[col] for col in Feature_Columns_List]))
    
#     # 确定需要哪些聚合类型
#     needs_sum = 'SUM' in Groupby_Type_List or 'AVG' in Groupby_Type_List
#     needs_count = 'COUNT' in Groupby_Type_List or 'AVG' in Groupby_Type_List
#     needs_min = 'MIN' in Groupby_Type_List
#     needs_max = 'MAX' in Groupby_Type_List

#     # 初始化值的长度
#     zero_value_length = (len(Feature_Columns_List) * (needs_sum + needs_min + needs_max)) + needs_count
#     zero_value = [0] * zero_value_length
#     if needs_min:
#         zero_value[len(Feature_Columns_List) * needs_sum : len(Feature_Columns_List) * (needs_sum + needs_min)] = [float('inf')] * len(Feature_Columns_List)
#     if needs_max:
#         zero_value[len(Feature_Columns_List) * (needs_sum + needs_min) : len(Feature_Columns_List) * (needs_sum + needs_min + needs_max)] = [float('-inf')] * len(Feature_Columns_List)

#     # 序列化函数
#     def seqOp(acc, value):
#         offset = 0
#         for i, v in enumerate(value):
#             if needs_sum:
#                 acc[offset + i] += v
#             if needs_min:
#                 acc[offset + len(Feature_Columns_List) * needs_sum + i] = min(acc[offset + len(Feature_Columns_List) * needs_sum + i], v)
#             if needs_max:
#                 acc[offset + len(Feature_Columns_List) * (needs_sum + needs_min) + i] = max(acc[offset + len(Feature_Columns_List) * (needs_sum + needs_min) + i], v)
#         if needs_count:
#             acc[-1] += 1
#         return acc

#     # 组合函数
#     def combOp(acc1, acc2):
#         offset = 0
#         for i in range(len(Feature_Columns_List)):
#             if needs_sum:
#                 acc1[offset + i] += acc2[offset + i]
#             if needs_min:
#                 acc1[offset + len(Feature_Columns_List) * needs_sum + i] = min(acc1[offset + len(Feature_Columns_List) * needs_sum + i], acc2[offset + len(Feature_Columns_List) * needs_sum + i])
#             if needs_max:
#                 acc1[offset + len(Feature_Columns_List) * (needs_sum + needs_min) + i] = max(acc1[offset + len(Feature_Columns_List) * (needs_sum + needs_min) + i], acc2[offset + len(Feature_Columns_List) * (needs_sum + needs_min) + i])
#         if needs_count:
#             acc1[-1] += acc2[-1]
#         return acc1

#     # 使用 aggregateByKey 进行聚合
#     aggregated_rdd = pair_rdd.aggregateByKey(zero_value, seqOp, combOp)

#     # 将结果转换回 DataFrame
#     def to_row(key, values):
#         row_dict = {Aim_Column_name: key}
#         offset = 0
#         for i, col in enumerate(Feature_Columns_List):
#             if needs_sum:
#                 row_dict[f"SUM_{col}"] = values[offset + i]
#             if needs_min:
#                 row_dict[f"MIN_{col}"] = values[offset + len(Feature_Columns_List) * needs_sum + i]
#             if needs_max:
#                 row_dict[f"MAX_{col}"] = values[offset + len(Feature_Columns_List) * (needs_sum + needs_min) + i]
#             if needs_count:
#                 row_dict["Groupby_COUNT"] = values[-1]
#             if 'AVG' in Groupby_Type_List:
#                 count = values[-1] if values[-1] > 0 else 1
#                 row_dict[f"AVG_{col}"] = values[offset + i] / count
#         return row_dict

#     result_df = Spark_Session.createDataFrame(aggregated_rdd.map(lambda x: to_row(x[0], x[1])))

#     return result_df

# """
# 作用：
#     针对指定的rdd数据，计算groupby后的结果

# 输入：
#     Spark_Session：pyspark接口
#     Aim_Table_Rdd：目标pyspark数据
#     Groupby_Column_List: 目标列列名
#     Aggregate_Columns_List：聚合列列名
#     Groupby_Type_List：目标groupby运算方式

# 返回值：
#     采样后的pyspark数据
# """
# def Groupby_Pyspark_Table(Spark_Session, Aim_Table_Rdd, Groupby_Column_List, Aggregate_Columns_List, Groupby_Type_List):
#     Aim_Table_Rdd.createOrReplaceTempView("EMP")
    
#     sql_str = "SELECT" 
#     for tmp_column_name in Groupby_Column_List:
#         sql_str = sql_str + ' ' + tmp_column_name + ','
    
#     for tmp_groupby_type in Groupby_Type_List:
#         if tmp_groupby_type in ['AVG', 'SUM', 'MAX', 'MIN']:
#             for tmp_feature_column in Aggregate_Columns_List:
#                 sql_str = (sql_str + ' ' + tmp_groupby_type + "(" + tmp_feature_column + ") as " + tmp_groupby_type  
#                         + '_' + tmp_feature_column + ',')
#         elif tmp_groupby_type == 'COUNT':
#             sql_str = (sql_str + ' ' + tmp_groupby_type + "(*) as Groupby_COUNT,")
    
#     # 删去最后的逗号
#     sql_str = sql_str[:-1]
    
#     sql_str = sql_str + " FROM EMP GROUP BY " + Groupby_Column_List[0]
    
#     for tmp_column_name in Groupby_Column_List[1:]:
#         sql_str = sql_str + ', ' + tmp_column_name
    
#     tmp_groupby_result = Spark_Session.sql(sql_str)
    
#     return tmp_groupby_result


"""
作用：
    根据范围，保留pyspark文件中的指定范围的行

输入：
    Spark_Session：pyspark接口
    tmp_aim_small_rdd：目标pyspark数据
    tmp_node_range_np：目标行范围

返回值：
    采样后的pyspark数据
"""
def sample_rdd_from_aim_row(Spark_Session, tmp_aim_small_rdd, tmp_node_range_np, show_info = True):
    # 加上临时id号
    w = Window().orderBy(lit('tmp_order_lit'))
    tmp_aim_small_rdd = tmp_aim_small_rdd.withColumn("tmp_id", row_number().over(w) - 1)

    if show_info:
        print('保留指定范围行:已完成临时id号的生成')
    
    # 将行号转化为rdd格式
    aim_tmp_id_rdd = Spark_Session.createDataFrame(pd.DataFrame({'tmp_id':tmp_node_range_np}),["tmp_id"])
    
    if show_info:
        print('生成目标id号表')

    # 通过join获取保留的行号(如果列数较少，就先broadcast再join)
    if tmp_node_range_np.shape[0] > 100000:
        tmp_sampled_aim_small_rdd = aim_tmp_id_rdd.join(tmp_aim_small_rdd, 'tmp_id', 'inner')
    else:
        tmp_sampled_aim_small_rdd = aim_tmp_id_rdd.join(broadcast(tmp_aim_small_rdd), 'tmp_id', 'inner')
        
    # 删去临时id号
    tmp_sampled_aim_small_rdd = tmp_sampled_aim_small_rdd.drop('tmp_id')
    
    return tmp_sampled_aim_small_rdd

"""
作用：
    根据范围，保留pyspark文件中的指定范围的行

输入：
    Spark_Session：pyspark接口
    tmp_aim_small_rdd：目标pyspark数据
    tmp_node_range_start：目标行起始范围
    tmp_node_range_end：目标行终止范围

返回值：
    采样后的pyspark数据
"""
def sample_rdd_from_aim_range(Spark_Session, tmp_aim_small_rdd, tmp_node_range_start, tmp_node_range_end, show_info = True):
    # index号得从低到高，且连续，且保证一致
    
    # 加上临时id号
    w = Window().orderBy(lit('tmp_order_lit'))
    tmp_aim_small_rdd = tmp_aim_small_rdd.withColumn("tmp_id", row_number().over(w) - 1)

    if show_info:
        print('保留指定范围行:已完成临时id号的生成')
    
    tmp_aim_small_rdd = tmp_aim_small_rdd.where((tmp_aim_small_rdd.tmp_id >= tmp_node_range_start) & 
                                  (tmp_aim_small_rdd.tmp_id < tmp_node_range_end))
        
    # 删去临时id号
    tmp_aim_small_rdd = tmp_aim_small_rdd.drop('tmp_id')
    
    return tmp_aim_small_rdd

"""
作用：
    随机采样pyspark数据中的n行（最好是小文件，因为非常耗时）

输入：
    Spark_Session：pyspark接口
    tmp_aim_small_rdd：目标pyspark数据
    tmp_sample_n_num：要随机采样的行数
    tmp_aim_small_rdd_count：目标pyspark数据的总行数（可以不设置，但就得算一遍count，较为耗时）

返回值：
    采样后的pyspark数据
"""
def sample_random_n_samples_for_samll_rdd(Spark_Session, tmp_aim_small_rdd, tmp_sample_n_num, tmp_aim_small_rdd_count = 0):
    # 加上临时id号
    w = Window().orderBy(lit('tmp_order_lit'))
    tmp_aim_small_rdd = tmp_aim_small_rdd.withColumn("tmp_id", row_number().over(w) - 1)

    print('随机采样n个样本任务:已完成临时id号的生成')
#     print('最大行号为:', tmp_aim_small_rdd.agg({'tmp_id': "max"}).collect()[0])

    if tmp_aim_small_rdd_count < 1:
        tmp_aim_small_rdd_count = tmp_aim_small_rdd.count()
    
    # 生成要选取的行号
    tmp_sample_ids = random.sample(range(0, tmp_aim_small_rdd_count), tmp_sample_n_num)
    tmp_sample_ids.sort()

    # 将行号转化为rdd格式
    aim_tmp_id_rdd = Spark_Session.createDataFrame(pd.DataFrame({'tmp_id':tmp_sample_ids}),["tmp_id"])
    print('生成目标id号表')

    # 通过join获取保留的行号
    tmp_sampled_aim_small_rdd = aim_tmp_id_rdd.join(tmp_aim_small_rdd, 'tmp_id', 'inner')

    # 删去临时id号
    tmp_sampled_aim_small_rdd = tmp_sampled_aim_small_rdd.drop('tmp_id')
    
    return tmp_sampled_aim_small_rdd

"""
作用：
    在pyspark中对每个样本groupby后再选取权重最高的n行（最好是小文件，因为非常耗时，如果有同权重的，则从中随机采样）

输入：
    Spark_Session：pyspark接口
    tmp_aim_small_rdd：目标pyspark数据
    tmp_aim_column_name: groupby目标列
    tmp_sample_n_num：要采样的行数
    tmp_Weight_Column_name:权重列列名
    random_sample_for_same_number:是否给相同的数值生成不同的序号

返回值：
    采样后的pyspark数据
"""
def sample_top_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_aim_small_rdd, tmp_aim_column_name, tmp_sample_n_num, tmp_Weight_Column_name,
                                random_sample_for_same_number = True):
    if random_sample_for_same_number:
        groupby_window = Window.partitionBy(tmp_aim_small_rdd[tmp_aim_column_name]).orderBy(col(tmp_Weight_Column_name).desc(), F.rand())
    else:
        groupby_window = Window.partitionBy(tmp_aim_small_rdd[tmp_aim_column_name]).orderBy(col(tmp_Weight_Column_name).desc())
    
    tmp_sampled_aim_small_rdd = tmp_aim_small_rdd.select('*', F.rank().over(groupby_window).alias('rank')).filter(F.col('rank') <= 
                                                                              tmp_sample_n_num).drop('rank')
    
    return tmp_sampled_aim_small_rdd


"""
作用：
    在pyspark中对每个样本groupby后再随机采样n行（最好是小文件，因为非常耗时）

输入：
    Spark_Session：pyspark接口
    tmp_aim_small_rdd：目标pyspark数据
    tmp_aim_column_name: groupby目标列
    tmp_sample_n_num：groupby后再随机采样的行数

返回值：
    采样后的pyspark数据
"""
def sample_random_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_aim_small_rdd, tmp_aim_column_name, tmp_sample_n_num):
    
    groupby_window = Window.partitionBy(tmp_aim_small_rdd[tmp_aim_column_name]).orderBy(F.rand())
    tmp_sampled_aim_small_rdd = tmp_aim_small_rdd.select('*', F.rank().over(groupby_window).alias('rank')).filter(F.col('rank') <= 
                                                                              tmp_sample_n_num).drop('rank')
    
    return tmp_sampled_aim_small_rdd


"""
作用：

输入：

返回值：
    预处理后的pyspark数据
"""
def Preprocess_Numerical_Data(Target_Pyspark_df, preprocess_type_list, Stable_Columns, Preprocess_Columns):
    # Fill null with 0
    Target_Pyspark_df = Target_Pyspark_df.fillna(0)
    
    # Prepare the pipeline stages
    pipeline_stages = []
    
    # Create a VectorAssembler to assemble the features into a single vector
    feature_assembler = VectorAssembler(inputCols=Preprocess_Columns, outputCol="Feature_Raw")
    pipeline_stages.append(feature_assembler)

    # If normalization is requested, add a MinMaxScaler to the pipeline
    if 'Normalized' in preprocess_type_list:
        minmax_scaler = MinMaxScaler(inputCol="Feature_Raw", outputCol="Feature_Normalized")
        pipeline_stages.append(minmax_scaler)
        
    # If standardization is requested, add a StandardScaler to the pipeline
    if 'Standard' in preprocess_type_list:
        standard_scaler = StandardScaler(inputCol="Feature_Raw", outputCol="Feature_Standard")
        pipeline_stages.append(standard_scaler)
        
    # Create and fit the pipeline
    pipeline = Pipeline(stages=pipeline_stages)
    model = pipeline.fit(Target_Pyspark_df)
    processed_df = model.transform(Target_Pyspark_df)
    
    # Split vectors into separate columns
    output_columns = Stable_Columns + ["Feature_" + x for x in preprocess_type_list]
            
    # Drop the intermediate feature vector columns
    processed_df = processed_df.select(output_columns)

    return processed_df

"""

"""
def Pyspark_String_to_Vector(s):
    if s.startswith("(") and "," in s and "[" in s:
        # 解析稀疏向量格式
        size, indices_str, values_str = re.match(r'\((\d+),\s*\[(.*?)\],\s*\[(.*?)\]\)', s).groups()
        size = int(size)
        indices = [int(x) for x in indices_str.split(",")]
        values = [float(x) for x in values_str.split(",")]
        return SparseVector(size, indices, values)
    elif s.startswith("["):
        # 提取数字作为密集向量
        values = [float(x) for x in re.findall(r'([-+]?\d*\.\d+|\d+)', s)]
        return DenseVector(values)
    else:
        raise ValueError("Unknown vector format")

"""
"""
def Pyspark_Merge_Vectors(Spark_Session, df, col1, col2):

    @udf(VectorUDT())
    def merge_vectors(vec1, vec2):
        # 如果两个向量都是SparseVector
        if isinstance(vec1, SparseVector) and isinstance(vec2, SparseVector):
            indices = list(vec1.indices) + list(vec2.indices + vec1.size)
            values = list(vec1.values) + list(vec2.values)
            size = vec1.size + vec2.size
            return SparseVector(size, indices, values)
        else:
            return Vectors.dense(vec1.toArray().tolist() + vec2.toArray().tolist())
    
    # 使用UDF合并向量
    merged_df = df.withColumn(col1, merge_vectors(col(col1), col(col2))).drop(col2)
    
    return merged_df
    
"""
"""
def Pyspark_Fill_Null_Vectors(Spark_Session, joined_df, column_names, vector_length = None):
    if vector_length is None:
        # 1. 选择第一个列名并过滤出非空值
        column_name = column_names[0]
        filtered_df = joined_df.select(column_name).filter(col(column_name).isNotNull())

        # 2. 使用limit(1)获取一个非空值
        limited_df = filtered_df.limit(1)

        # 3. 从结果DataFrame中获取向量长度
        results = limited_df.collect()
        if results:
            vector_value = results[0][0]
            vector_length = len(vector_value) if vector_value else 0
        else:
            vector_length = None
    
    if vector_length is not None:
        # 如果存在非null向量，使用该长度为 null 向量列填充一个空的向量
        empty_vector = SparseVector(vector_length, {})

        # 创建一个UDF来填充空向量
        fill_udf = udf(lambda v: v if v is not None else empty_vector, VectorUDT())

        for col_name in column_names:
            joined_df = joined_df.withColumn(col_name, fill_udf(col(col_name)))
    else:
        print("Error: All the rows are none")
    
    return joined_df


# def hdfs_list_files(hdfs_path):
#     fs = pa.hdfs.connect()
#     tmp_result = fs.ls(hdfs_path)
#     fs.close()
#     return tmp_result

# def hdfs_file_exists(hdfs_path):
#     fs = pa.hdfs.connect()
#     tmp_result = fs.exists(hdfs_path)
#     fs.close()
#     return tmp_result

# def hdfs_create_dir(hdfs_path):
#     # 连接到 HDFS
#     fs = pa.hdfs.connect()
    
#     try:
#         fs.mkdir(hdfs_path)
#         print(f'Create dir at {hdfs_path} ')
#     except Exception as e:
#         print(f'Failed to create dir: {e}')
#     finally:
#         # 关闭 HDFS 连接
#         fs.close()

# def hdfs_create_marker_file(hdfs_path, file_name='_SUCCESS_Marker'):
#     # 连接到 HDFS
#     fs = pa.hdfs.connect()
    
#     marker_file_path = hdfs_path + '/' + file_name
    
#     try:
#         # 创建并写入文件
#         with fs.open(marker_file_path, 'wb') as f:
#             # 获取当前时间并写入文件
#             timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             f.write(("Write operation completed successfully at " + timestamp).encode('utf-8'))
#         print(f'SUCCESS_Marker.txt file created at {marker_file_path} with timestamp {timestamp}')
#     except Exception as e:
#         print(f'Failed to create marker file: {e}')
#     finally:
#         # 关闭 HDFS 连接
#         fs.close()

def hdfs_read_txt_file(file_path):
    hdfs = pafs.HadoopFileSystem(host="default")
    # 检查文件是否存在
    file_info = hdfs.get_file_info(file_path)
    if file_info.type != pafs.FileType.NotFound:
        with hdfs.open_input_stream(file_path) as file:
            # 读取文件内容
            content = file.read()
            # 将字节内容解码为 UTF-8 格式的字符串
            return content.decode('utf-8')
    else:
        return "文件不存在"

def hdfs_list_files(hdfs_path):
    hdfs = pafs.HadoopFileSystem(host = "default")
    # 获取目录下的文件信息
    file_info_list = hdfs.get_file_info(pafs.FileSelector(hdfs_path, recursive=False))
    # 提取文件路径并返回字符串列表
    return [info.path for info in file_info_list if info.type != pafs.FileType.NotFound]

def hdfs_file_exists(hdfs_path):
    hdfs = pafs.HadoopFileSystem(host = "default")
    return hdfs.get_file_info(hdfs_path).type != pafs.FileType.NotFound

def hdfs_create_dir(hdfs_path):
    hdfs = pafs.HadoopFileSystem(host = "default")
    try:
        hdfs.create_dir(hdfs_path)
        print(f'Create dir at {hdfs_path}')
    except Exception as e:
        print(f'Failed to create dir: {e}')

def hdfs_create_marker_file(hdfs_path, file_name='_SUCCESS_Marker', content = None):
    hdfs = pafs.HadoopFileSystem(host = "default")
    marker_file_path = hdfs_path + '/' + file_name
    
    try:
        with hdfs.open_output_stream(marker_file_path) as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if content is None:
                f.write(("Write operation completed successfully at " + timestamp).encode('utf-8'))
                print(f'{file_name} file created at {marker_file_path} with timestamp {timestamp}')
            else:
                f.write(content.encode('utf-8'))
                print(f'{file_name} file created at {marker_file_path} with content {content:.500}')
    except Exception as e:
        print(f'Failed to create marker file: {e}')

def hdfs_read_marker_file(hdfs_path, file_name='_SUCCESS_Marker'):
    hdfs = pafs.HadoopFileSystem(host = "default")
    marker_file_path = hdfs_path + '/' + file_name

    try:
        with hdfs.open_input_stream(marker_file_path) as f:
            return f.read().decode('utf-8')
    except Exception as e:
        print(f'Failed to read marker file: {e}')
        return None
        
def create_marker_file(spark, hdfs_path, file_name='_SUCCESS_Marker'):
    """
    Create a marker file in HDFS with a timestamp.
    
    Args:
    - spark (SparkSession): Active SparkSession.
    - hdfs_path (str): Path in HDFS where the marker file should be created.
    - file_name (str, optional): Name of the marker file. Defaults to '_SUCCESS_Marker'.
    
    Returns:
    - None
    """
    # 导入需要的 Java 类
    java_import(spark._jvm, 'org.apache.hadoop.fs.Path')
    java_import(spark._jvm, 'org.apache.hadoop.fs.FileSystem')
    
    # 获取Hadoop FileSystem实例
    fs = spark._jvm.FileSystem.get(spark._jsc.hadoopConfiguration())
    
    # 定义要创建和写入的文件的路径
    marker_file_path = hdfs_path + '/' + file_name
    
    try:
        # 创建标记文件
        file_stream = fs.create(spark._jvm.Path(marker_file_path))
        
        # 获取当前时间并写入文件
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_stream.write(("Write operation completed successfully at " + timestamp).encode('utf-8'))
        
        # 关闭文件流
        file_stream.close()
        
        print(f'SUCCESS_Marker.txt file created at {marker_file_path} with timestamp {timestamp}')
    
    except Exception as e:
        print(f"Error creating marker file: {e}")
    finally:
        # Ensure resources are closed properly
        file_stream.close()

def Pyspark_Left_Join_and_Fill_Null_Vectors(Spark_Session, Left_Table, Right_Table, Key_Columns, Vector_Columns, vector_length = None):
    # Perform an initial left join
    joined_df = Left_Table.join(Right_Table, Key_Columns, "left")
    
    # Filter out the DataFrame into two parts
    joined_df_value = joined_df.filter(col(Vector_Columns[0]).isNotNull())
    joined_df_null = joined_df.filter(col(Vector_Columns[0]).isNull())

    joined_df_value = joined_df_value.repartition(*Key_Columns)
    joined_df_null = joined_df_null.repartition(*Key_Columns)
    
    if vector_length is None:
        # Determine the vector length from the right table
        vec_col = Vector_Columns[0]
        vector_value = Right_Table.select(vec_col).filter(col(vec_col).isNotNull()).limit(1).collect()
        if vector_value:
            vector_length = len(vector_value[0][0])
            print("Vector length:", vector_length)
        else:
            raise ValueError("Cannot determine vector length from the right table")
    else:
        print("Vector length:", vector_length)
        
    # Create the default vector
    default_vector = SparseVector(vector_length, {})
    
    # 创建一个UDF来填充空向量
    fill_udf = udf(lambda v: default_vector, VectorUDT())

    for col_name in Vector_Columns:
        joined_df_null = joined_df_null.withColumn(col_name, fill_udf(col(col_name)))
    
#     # Create a DataFrame with a single row of default vectors
#     default_row = [(default_vector,) * len(Vector_Columns)]
#     default_vectors_df = Spark_Session.createDataFrame(default_row, Vector_Columns)
    
#     # Add a unique ID column set to 0 to both DataFrames
#     joined_df_null = joined_df_null.withColumn("unique_id", lit(0))
#     joined_df_null = joined_df_null.repartition(*Key_Columns)
    
#     default_vectors_df = default_vectors_df.withColumn("unique_id", lit(0))
    
#     # Drop Vector_Columns from joined_df_null and join with default_vectors_df
#     for col_name in Vector_Columns:
#         joined_df_null = joined_df_null.drop(col_name)
#     joined_df_null = joined_df_null.join(default_vectors_df, "unique_id", "left").drop("unique_id")

    # Concatenate the value and null DataFrames
    final_df = joined_df_value.unionByName(joined_df_null.select(joined_df_value.columns))

    return final_df

# Keep the numerical columns and the corresponding comment
def check_numeric_columns(df_schema):
    numeric_types = (IntegerType, FloatType, DoubleType, DecimalType, LongType, ShortType, ByteType)

    numeric_columns = []
    numeric_columns_comment = []
    for field in df_schema.fields:
        if isinstance(field.dataType, numeric_types):
            numeric_columns.append(field.name)
            numeric_columns_comment.append(field.metadata.get('comment', field.name) if 'comment' in field.metadata else field.name)
    return numeric_columns, numeric_columns_comment

def estimate_row_size(df):
    """
    根据 DataFrame 的列类型估算每行数据的大小。

    :param df: DataFrame
    :return: 每行数据的估计大小（字节）
    """
    # 定义不同数据类型的估计大小
    size_estimates = {
        'IntegerType': 4,    # 整型
        'LongType': 8,       # 长整型
        'FloatType': 4,      # 浮点型
        'DoubleType': 8,     # 双精度浮点型
        'StringType': 20,    # 字符串型，平均估计值
        # 根据需要添加更多类型
    }

    # 计算每行的估计大小
    row_size = sum(size_estimates.get(field.dataType.simpleString(), 16)  # 使用默认值作为未知类型的大小
                  for field in df.schema.fields)

    return row_size

def estimate_partitions(df, partition_target_size_mb=300, row_size_estimate = None, approximate_row_count = None):
    """
    估算所需的分区数。

    :param df: 要估算的 DataFrame
    :param partition_target_size_mb: 每个分区的目标大小（MB）
    :return: 估算的分区数
    """
    # 转换 MB 为字节
    partition_target_size_bytes = partition_target_size_mb * 1024 * 1024

    # 估算每行数据的大小
    if row_size_estimate is None:
        row_size_estimate = estimate_row_size(df)

    # 使用 countApprox 获取近似行数
    if approximate_row_count is None:
        approximate_row_count = df.rdd.countApprox(timeout=1000, confidence=0.95)
        
    # 估算总数据大小
    estimated_total_size_bytes = approximate_row_count * row_size_estimate

    # 计算所需的分区数
    num_partitions = int(math.ceil(estimated_total_size_bytes / partition_target_size_bytes))

    return num_partitions

def Spark_Random_N_Sample(Spark_Session, target_df, target_node_columns, max_count):
    
    groupby_window = Window.partitionBy(*[target_df[c] for c in target_node_columns]).orderBy(F.rand())
    
    tmp_filtered_df = target_df.select('*', F.row_number().over(groupby_window).alias('row_number'))
    
    tmp_filtered_df = tmp_filtered_df.filter(F.col('row_number') <= max_count).drop('row_number')
    
    return tmp_filtered_df

def Spark_Top_N_Sample(Spark_Session, target_df, target_node_columns, max_count, target_feat_columns):
    # 构建排序条件
    order_columns = [col(c).desc() for c in target_feat_columns]
    order_columns.append(F.rand())

    groupby_window = Window.partitionBy(*[target_df[c] for c in target_node_columns]).orderBy(*order_columns)
    
    tmp_filtered_df = target_df.select('*', F.row_number().over(groupby_window).alias('row_number'))
    
    tmp_filtered_df = tmp_filtered_df.filter(F.col('row_number') <= max_count).drop('row_number')
    
    return tmp_filtered_df

# def Spark_Threshold_N_Sample(Spark_Session, target_df, target_node_columns, max_count):
#     # 定义窗口，按照指定列进行分组
#     groupby_window = Window.partitionBy(*[target_df[c] for c in target_node_columns])

#     # 计算每个组的行数(不该直接用groupby的，待优化*)
#     tmp_filtered_df = target_df.withColumn('group_count', F.count('*').over(groupby_window))

#     # 过滤掉行数超过 max_count 的组
#     tmp_filtered_df = tmp_filtered_df.filter(F.col('group_count') <= max_count).drop('group_count')
    
#     return tmp_filtered_df

def Spark_Threshold_N_Sample(spark, target_df, target_node_columns, max_count):
    # 将DataFrame转换为RDD
    rdd = target_df.rdd

    # 生成Key-Value对，其中Key是分组列的组合，Value是行数据
    key_value_rdd = rdd.map(lambda row: (tuple(row[c] for c in target_node_columns), row))

    # 使用aggregateByKey来过滤掉行数超过max_count的组
    aggregated_rdd = key_value_rdd.aggregateByKey((0, []), 
                                   lambda acc, value: (acc[0] + 1, acc[1] + [value]) if acc[0] < max_count else (acc[0] + 1, acc[1]),
                                   lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]) if acc1[0] + acc2[0] <= max_count else (acc1[0] + acc2[0], []))

    # 过滤掉超过max_count的组，然后将结果RDD转换回DataFrame
    result_df = aggregated_rdd.filter(lambda x: len(x[1][1]) <= max_count).flatMap(lambda x: x[1][1]).map(lambda row: Row(**row.asDict())).toDF()

    return result_df

def sanitize_column_names(df):
    """
    检测 DataFrame 的列名中是否包含特殊字符，并只将这些字符替换为下划线。

    :param df: 原始的 PySpark DataFrame
    :return: 列名被清理后的 DataFrame
    """
    # 定义一个函数，用于检测并替换列名中的特殊字符
    def replace_special_chars(col_name):
        # 定义特殊字符的正则表达式
        # 这里假设特殊字符是除了字母、数字和下划线之外的任何字符
        pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9_]'
        # 检测是否有特殊字符
        if re.search(pattern, col_name):
            # 将匹配的特殊字符替换为下划线
            return re.sub(pattern, '_', col_name)
        else:
            # 没有特殊字符，返回原始列名
            return col_name

    # 对每个列名应用替换函数
    new_columns = [(col, replace_special_chars(col)) for col in df.columns]

    # 使用新的列名创建一个新的 DataFrame
    for old_col, new_col in new_columns:
        if old_col != new_col:
            print('对不规范的列名进行替换:', old_col, new_col)
            df = df.withColumnRenamed(old_col, new_col)

    return df