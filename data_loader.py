"""
data_loader.py - 数据加载模块
"""
import pandas as pd
from config import Config

def load_and_preprocess(cfg=None):

    if cfg is None:
        cfg = Config()
    
    # 读取Excel文件
    try:
        df = pd.read_excel(cfg.DATA_PATH, header=1)
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件不存在: {cfg.DATA_PATH}")
    except Exception as e:
        raise RuntimeError(f"读取数据文件失败: {e}")
    
    # 获取所有列名
    cols = df.columns.tolist()
    
    # 找出南北侧温度列（通过列名模糊匹配）
    south_cols = [col for col in cols if '风箱南侧温度' in col]
    north_cols = [col for col in cols if '风箱北侧温度' in col]
    
    # 按风箱编号排序（假设列名格式为 '1#风箱南侧温度'）
    south_cols.sort(key=lambda x: int(x.split('#')[0]))
    north_cols.sort(key=lambda x: int(x.split('#')[0]))
    
    # 合并温度
    for i in range(1, 19):
        # 对应列可能不存在（数据缺失），但根据题意应当存在
        if i-1 >= len(south_cols) or i-1 >= len(north_cols):
            raise ValueError(f"缺少风箱{i}的南北侧温度列")
        
        south = south_cols[i-1]
        north = north_cols[i-1]
        temp_col = f'{i}#风箱温度'
        
        # 转换为数值（处理可能的字符串或空值）
        s_val = pd.to_numeric(df[south], errors='coerce')
        n_val = pd.to_numeric(df[north], errors='coerce')
        df[temp_col] = (s_val + n_val) / 2
    
    # 删除含有缺失值的行（如果有）
    df = df.dropna().reset_index(drop=True)
    
    return df

def get_train_test_split(df, train_ratio=None):
    """
    按时间顺序划分训练集和测试集
    """
    if train_ratio is None:
        cfg = Config()
        train_ratio = cfg.TRAIN_SPLIT
    
    train_size = int(train_ratio * len(df))
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()
    return train, test