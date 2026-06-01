"""
feature_engineering.py - 特征工程模块
"""
import pandas as pd
import numpy as np

def add_lag_features(df, col, lags, suffix='lag'):
    """
    为DataFrame的指定列添加多个滞后特征
    
    Parameters
    ----------
    df : pd.DataFrame
    col : str
        需要添加滞后的列名
    lags : list of int
        滞后步数列表
    suffix : str
        滞后特征后缀，默认为 'lag'
    
    Returns
    -------
    pd.DataFrame
        添加滞后列后的DataFrame（原地修改，同时返回）
    """
    for lag in lags:
        df[f'{col}_{suffix}{lag}'] = df[col].shift(lag)
    return df


def add_rolling_stats(df, col, windows, stats, min_periods=1):
    """
    为DataFrame的指定列添加滑动窗口统计特征
    
    Parameters
    ----------
    df : pd.DataFrame
    col : str
        原始列名
    windows : list of int
        窗口大小列表
    stats : list of str
        统计量列表，支持 'mean', 'std', 'min', 'max', 'range', 'slope'
    min_periods : int
        最小周期数，默认1
    
    Returns
    -------
    pd.DataFrame
        添加统计特征后的DataFrame（原地修改）
    """
    for w in windows:
        rolling = df[col].rolling(window=w, min_periods=min_periods)
        
        if 'mean' in stats:
            df[f'{col}_win_mean_{w}'] = rolling.mean()
        if 'std' in stats:
            df[f'{col}_win_std_{w}'] = rolling.std()
        if 'min' in stats:
            df[f'{col}_win_min_{w}'] = rolling.min()
        if 'max' in stats:
            df[f'{col}_win_max_{w}'] = rolling.max()
        if 'range' in stats:
            df[f'{col}_win_range_{w}'] = rolling.max() - rolling.min()
        if 'slope' in stats:
            # 使用窗口内首尾差值除以步数作为斜率近似
            def _slope(arr):
                if len(arr) <= 1:
                    return 0.0
                return (arr.iloc[-1] - arr.iloc[0]) / (len(arr) - 1)
            df[f'{col}_win_slope_{w}'] = rolling.apply(_slope, raw=False)
    
    return df


def add_diff_features(df, col, periods, suffix='diff'):
    """
    添加差分特征
    
    Parameters
    ----------
    df : pd.DataFrame
    col : str
        原始列名
    periods : list of int
        差分步数列表
    suffix : str
        差分特征后缀，默认 'diff'
    
    Returns
    -------
    pd.DataFrame
    """
    for p in periods:
        df[f'{col}_{suffix}{p}'] = df[col].diff(p)
    return df


def build_fan_features_for_one_fan(fan_idx, train, test, cfg):
    """
    为单个风箱构建训练和测试的特征矩阵及目标值
    
    Parameters
    ----------
    fan_idx : int
        风箱编号 (1~18)
    train : pd.DataFrame
        训练集原始数据
    test : pd.DataFrame
        测试集原始数据
    cfg : Config
        配置对象
    
    Returns
    -------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    """
    pressure_col = f'{fan_idx}#风箱负压'
    target_col = f'{fan_idx}#风箱温度'
    
    N_LAGS = cfg.FAN_N_LAGS
    NEIGHBORS = cfg.FAN_NEIGHBORS
    WINDOW = cfg.FAN_WINDOW
    GLOBAL_FEATURES = cfg.GLOBAL_FEATURES
    GLOBAL_LAGS = cfg.FAN_GLOBAL_LAGS
    
    # 自身滞后特征
    lag_cols = [f'{pressure_col}_lag{l}' for l in range(1, N_LAGS+1)]
    
    # 相邻风箱负压（当前）
    neighbor_pressure = []
    for d in range(-NEIGHBORS, NEIGHBORS+1):
        if d == 0: continue
        nf = fan_idx + d
        if 1 <= nf <= 18:
            neighbor_pressure.append(f'{nf}#风箱负压')
    
    # 相邻风箱温度（当前）
    neighbor_temp = []
    for d in range(-NEIGHBORS, NEIGHBORS+1):
        if d == 0: continue
        nf = fan_idx + d
        if 1 <= nf <= 18:
            neighbor_temp.append(f'{nf}#风箱温度')
    
    # 相邻风箱温度滞后
    neighbor_temp_lag = []
    if fan_idx == 5 or fan_idx >= 11:
        for d in [-1, 1]:
            nf = fan_idx + d
            if 1 <= nf <= 18:
                for lag in [5, 10, 20, 30]:
                    neighbor_temp_lag.append(f'{nf}#风箱温度_lag{lag}')
    else:
        for d in [-1, 1]:
            nf = fan_idx + d
            if 1 <= nf <= 18:
                for lag in [5, 10, 15]:
                    neighbor_temp_lag.append(f'{nf}#风箱温度_lag{lag}')
    
    # 窗口统计特征
    stat_cols = [f'{pressure_col}_window{WINDOW}_{stat}' for stat in ['mean', 'std', 'slope']]
    
    # 全局特征滞后
    global_lag_cols = []
    for gcol in GLOBAL_FEATURES:
        for lag in GLOBAL_LAGS:
            global_lag_cols.append(f'{gcol}_lag{lag}')
    
    # 组合基础特征
    feature_cols = ([pressure_col] + lag_cols + neighbor_pressure + neighbor_temp +
                    neighbor_temp_lag + stat_cols + GLOBAL_FEATURES + global_lag_cols)
    
    # 尾部增强
    if fan_idx >= 11:
        upstream_temp = []
        for u in range(max(1, fan_idx-3), fan_idx):
            upstream_temp.append(f'{u}#风箱温度')
            for lag in [5, 10, 20]:
                upstream_temp.append(f'{u}#风箱温度_lag{lag}')
        feature_cols += upstream_temp
        self_temp_lag = [f'{target_col}_lag{t}' for t in range(1, 21)]
        feature_cols += self_temp_lag
    elif fan_idx == 5:
        upstream_temp = [f'{u}#风箱温度' for u in [3,4]]
        feature_cols += upstream_temp
        self_temp_lag = [f'{target_col}_lag{t}' for t in range(1, 11)]
        feature_cols += self_temp_lag
    
    # ----- 构造所有滞后特征（临时副本，避免污染原数据）-----
    train_temp = train.copy()
    test_temp = test.copy()
    
    # 1. 自身负压滞后
    for l, col in zip(range(1, N_LAGS+1), lag_cols):
        train_temp[col] = train_temp[pressure_col].shift(l)
        test_temp[col] = test_temp[pressure_col].shift(l)
    
    # 2. 相邻风箱温度滞后
    for col in neighbor_temp_lag:
        parts = col.split('_lag')
        src_col = parts[0]
        lag = int(parts[1])
        train_temp[col] = train_temp[src_col].shift(lag)
        test_temp[col] = test_temp[src_col].shift(lag)
    
    # 3. 全局滞后
    for col in global_lag_cols:
        parts = col.split('_lag')
        src_col = parts[0]
        lag = int(parts[1])
        train_temp[col] = train_temp[src_col].shift(lag)
        test_temp[col] = test_temp[src_col].shift(lag)
    
    # 4. 窗口统计
    for stat in ['mean', 'std', 'slope']:
        rolling_train = train_temp[pressure_col].rolling(window=WINDOW, min_periods=1)
        if stat == 'mean':
            vals_train = rolling_train.mean()
            vals_test = test_temp[pressure_col].rolling(window=WINDOW, min_periods=1).mean()
        elif stat == 'std':
            vals_train = rolling_train.std()
            vals_test = test_temp[pressure_col].rolling(window=WINDOW, min_periods=1).std()
        else:  # slope
            def _slope(arr):
                if len(arr) <= 1:
                    return 0
                return (arr.iloc[-1] - arr.iloc[0]) / (len(arr)-1)
            vals_train = rolling_train.apply(_slope, raw=False)
            vals_test = test_temp[pressure_col].rolling(window=WINDOW, min_periods=1).apply(_slope, raw=False)
        train_temp[f'{pressure_col}_window{WINDOW}_{stat}'] = vals_train
        test_temp[f'{pressure_col}_window{WINDOW}_{stat}'] = vals_test
    
    # 5. 自身温度滞后（仅对5#和尾部）
    if fan_idx == 5 or fan_idx >= 11:
        max_lag = 20 if fan_idx >= 11 else 10
        for t in range(1, max_lag+1):
            col = f'{target_col}_lag{t}'
            train_temp[col] = train_temp[target_col].shift(t)
            test_temp[col] = test_temp[target_col].shift(t)
    
    # 6. 上游温度滞后（仅对5#和尾部）
    if fan_idx == 5:
        for u in [3,4]:
            for lag in [1,5,10]:
                col = f'{u}#风箱温度_lag{lag}'
                train_temp[col] = train_temp[f'{u}#风箱温度'].shift(lag)
                test_temp[col] = test_temp[f'{u}#风箱温度'].shift(lag)
                if col not in feature_cols:
                    feature_cols.append(col)
    elif fan_idx >= 11:
        for u in range(max(1, fan_idx-3), fan_idx):
            for lag in [5,10,20]:
                col = f'{u}#风箱温度_lag{lag}'
                if col not in feature_cols:
                    feature_cols.append(col)
                train_temp[col] = train_temp[f'{u}#风箱温度'].shift(lag)
                test_temp[col] = test_temp[f'{u}#风箱温度'].shift(lag)
    
    # 单独优化1#风箱：
    if fan_idx == 1:
        # 1. 增加点火区特有的上游信息（虽然没有上游风箱，但可增加机速与大烟道温度的长滞后）
        speed_lags = [5, 10, 20, 30, 40, 50, 60]
        for lag in speed_lags:
            feature_cols.append(f'烧结机机速L1设定_lag{lag}')
        stack_temp_lags = [10, 20, 30, 40, 50, 60]
        for lag in stack_temp_lags:
            feature_cols.append(f'1#大烟道温度_lag{lag}')
            feature_cols.append(f'2#大烟道温度_lag{lag}')
        # 2. 增加自身负压的更长期滞后（90步，180秒）
        extra_lags = [61, 70, 80, 90]
        for lag in extra_lags:
            feature_cols.append(f'{pressure_col}_lag{lag}')
        # 3. 增加自身负压的近期趋势（过去5步、10步的均值和斜率）
        for w in [5, 10]:
            feature_cols.append(f'{pressure_col}_win_mean_{w}')
            feature_cols.append(f'{pressure_col}_win_slope_{w}')
    
    # 删除缺失值
    train_clean = train_temp.dropna(subset=feature_cols).copy()
    test_clean = test_temp.dropna(subset=feature_cols).copy()
    
    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values
    X_test = test_clean[feature_cols].values
    y_test = test_clean[target_col].values
    
    return X_train, y_train, X_test, y_test