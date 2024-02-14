import os
import json
import sys

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

"""
作用：
    同时保存print的信息到文件

输入：
    file：保存信息的文件

返回值：
    无
"""
class Log_save(object):
    def __init__(self, *files):
        self.files = files
        
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
            
    def flush(self) :
        for f in self.files:
            f.flush()

"""
作用：
    同时保存print的信息到文件

输入：
    file：保存信息的文件

返回值：
    无
"""
class OutputLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Flush method needed for compatibility
        self.terminal.flush()
        self.log.flush()
            
"""
作用：
    根据给定的路径创建新文件夹

输入：
    path：要创建的文件夹对应的路径

返回值：
    无
"""
def mkdir(path):
    folder = os.path.exists(path)
    # 判断是否存在文件夹如果不存在则创建为文件夹
    if not folder:
        # makedirs 创建文件时如果路径不存在会创建这个路径
        os.makedirs(path)
        
"""
作用：
    根据给定的config文件路径读取配置信息，并返回字典类型的配置信息

输入：
    config_filename：目标config文件路径

返回值：
    字典类型的配置信息
"""
def read_json_config_file(config_filename):
    with open(config_filename, 'r') as f:
        config = json.loads(f.read())
    
    # 显示读取结果
    # print(json.dumps(config, sort_keys=True, indent=4))
    
    return config

"""
作用：
    根据给定的目标时间，将其切分为按月分隔的时间区间

输入：
    KG_time_range_list：给定的目标时间

返回值：
    按月分隔的时间区间
"""
def divid_range_list_to_monthly_list(KG_time_range_list):
    KG_time_monthly_list = []
    
    tmp_range_time = KG_time_range_list[0]
    tmp_range_time_add_month = tmp_range_time + relativedelta(months = 1)
    tmp_range_time_add_month = datetime(tmp_range_time_add_month.year, tmp_range_time_add_month.month, 1)
    
    while tmp_range_time_add_month < KG_time_range_list[1]:

        KG_time_monthly_list.append([tmp_range_time.strftime("%Y-%m-%d"), tmp_range_time_add_month.strftime("%Y-%m-%d")])
        
        tmp_range_time = tmp_range_time_add_month
        tmp_range_time_add_month = tmp_range_time + relativedelta(months = 1)
        
    KG_time_monthly_list.append([tmp_range_time.strftime("%Y-%m-%d"), KG_time_range_list[1].strftime("%Y-%m-%d")])
        
    return KG_time_monthly_list

"""
作用：
    根据给定的目标时间，将其切分为按月分隔的时间区间

输入：
    KG_time_range_list：给定的目标时间

返回值：
    按月分隔的时间区间
"""
def divid_range_list_to_monthly_first_day_list(KG_time_range_list):
    KG_time_monthly_list = []
    
    tmp_range_time = KG_time_range_list[0]
    
#     tmp_Label_Time_monthly_dt = datetime.strptime(tmp_Label_Time_monthly, "%Y-%m-%d")
    KG_time_monthly_list.append(datetime(tmp_range_time.year, tmp_range_time.month, 1).strftime("%Y-%m-%d"))
    
    tmp_range_time_add_month = tmp_range_time + relativedelta(months = 1)
    tmp_range_time_add_month = datetime(tmp_range_time_add_month.year, tmp_range_time_add_month.month, 1)
    
    while tmp_range_time_add_month < KG_time_range_list[1]:

        KG_time_monthly_list.append(tmp_range_time_add_month.strftime("%Y-%m-%d"))
        
        tmp_range_time = tmp_range_time_add_month
        tmp_range_time_add_month = tmp_range_time + relativedelta(months = 1)
        
    return KG_time_monthly_list