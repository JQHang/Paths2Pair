#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import io
import copy
import re
import gc
import json
import os
import sys
import math
import random
import time
import numpy as np
import pandas as pd

from random import sample
from collections import defaultdict
from tqdm import tqdm
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from py4j.protocol import Py4JError, Py4JJavaError, Py4JNetworkError

# Self packages path
Src_Dir_path = "../../.."
sys.path.append(Src_Dir_path)

# Self packages
from Utils.Pyspark_utils import ResilientSparkRunner
from Utils.utils import read_json_config_file
from Utils.utils import mkdir
from Utils.utils import OutputLogger


# In[2]:


#显示所有列
pd.set_option('display.max_columns', None)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

# 固定随机值
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
setup_seed(42)


# In[3]:


Data_In_HDFS_Path = '/user/mart_coo/mart_coo_innov/Complex_Path_Neighbor_Sampling/01_24_test'

# 本地数据输出路径
Data_Output_path = Src_Dir_path + "/.."

# 日志信息保存文件名
mkdir(Data_Output_path + '/Log/')
notebook_file_name = "0_ComplexPath_Recall_Data_Prepare"
log_file_name = Data_Output_path + f'/Log/{notebook_file_name}-' + datetime.now().strftime("%Y-%m-%d-%H:%M") + '.txt'

print('日志输出文件名:', log_file_name)

# Complex path config
ComplexPath_Config_file = '../Config/ComplexPath_Node_Recall_Config/Recall_Node_Config.json'
Node_Config_file = '../Config/Node_Config/Node_Config_Paths2Pairs_12_29.json'

ComplexPath_Config_dict = read_json_config_file(ComplexPath_Config_file)
Node_Config_dict = read_json_config_file(Node_Config_file)


# In[4]:


from Utils.Complex_Path_Basic_PySpark import complex_path_sampling
from Utils.Complex_Path_Basic_PySpark import complex_path_aggregation
from Utils.Decorator import Time_Costing

# 生成输出文件，并开始记录信息
sys.stdout = OutputLogger(log_file_name)
Spark_Runner = ResilientSparkRunner()

print('使用的复杂路配置:', ComplexPath_Config_file)

@Time_Costing
def Complexpath_Node_Recall(spark):
    # 目标时间
    target_times = ['2023-10-01']
    
    start_time = datetime.now()
    print(start_time)
    
    # 依次处理各个元路径
    for complex_path_index, complex_path_name in enumerate(ComplexPath_Config_dict):
        print('开始处理复杂路', complex_path_name)

        # 读取该元路径对应的特征并上传
        complex_path_aggregation(spark, complex_path_name, ComplexPath_Config_dict[complex_path_name], Node_Config_dict, target_times, 
                                 target_times, Data_In_HDFS_Path, target_node = None, feat_cols_output_type = 'columns')
        
        for (tmp_id, tmp_rdd) in spark.sparkContext._jsc.getPersistentRDDs().items():
            tmp_rdd.unpersist()
        
    end_time = datetime.now()
    print(end_time)
    
    return

Spark_Runner.run(Complexpath_Node_Recall)


# In[ ]:




