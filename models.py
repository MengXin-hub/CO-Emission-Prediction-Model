"""
model.py - 模型训练模块
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from config import Config
import xgboost as xgb

def train_fan_models(df, cfg=None):
    """
    训练每个风箱的温度预测模型（问题一）
    
    Parameters
    ----------
    df : pd.DataFrame
        已合并南北侧温度的数据
    cfg : Config, optional
        配置对象
    
    Returns
    -------
    dict: {
        'models': list of LGBMRegressor,
        'mae_list': list,
        'rmse_list': list,
        'r2_list': list,
        'y_tests': list,
        'y_preds': list
    }
    """
    if cfg is None:
        cfg = Config()
    
    # 划分训练集和测试集
    train_size = int(cfg.TRAIN_SPLIT * len(df))
    train = df.iloc[:train_size].copy()
    test = df.iloc[train_size:].copy()
    # print(f"训练集样本数: {len(train)}，测试集样本数: {len(test)}")
    
    # 超参数
    N_LAGS = cfg.FAN_N_LAGS
    NEIGHBORS = cfg.FAN_NEIGHBORS
    WINDOW = cfg.FAN_WINDOW
    GLOBAL_FEATURES = cfg.GLOBAL_FEATURES
    GLOBAL_LAGS = cfg.FAN_GLOBAL_LAGS
    
    scalers = []
    models = []
    mae_list, rmse_list, r2_list = [], [], []
    y_tests, y_preds = [], []
    test_pressures = []
    
    for i in range(1, 19):
        pressure_col = f'{i}#风箱负压'
        target_col = f'{i}#风箱温度'
        
        # ----- 特征构造（与pass3.py保持一致）-----
        # 自身滞后特征
        lag_cols = [f'{pressure_col}_lag{l}' for l in range(1, N_LAGS+1)]
        # 相邻风箱负压（当前）
        neighbor_pressure = []
        for d in range(-NEIGHBORS, NEIGHBORS+1):
            if d == 0: continue
            nf = i + d
            if 1 <= nf <= 18:
                neighbor_pressure.append(f'{nf}#风箱负压')
        # 相邻风箱温度（当前）
        neighbor_temp = []
        for d in range(-NEIGHBORS, NEIGHBORS+1):
            if d == 0: continue
            nf = i + d
            if 1 <= nf <= 18:
                neighbor_temp.append(f'{nf}#风箱温度')
        
        # 相邻风箱温度滞后
        neighbor_temp_lag = []
        if i == 5 or i >= 11:
            for d in [-1, 1]:
                nf = i + d
                if 1 <= nf <= 18:
                    for lag in [5, 10, 20, 30]:
                        neighbor_temp_lag.append(f'{nf}#风箱温度_lag{lag}')
        else:
            for d in [-1, 1]:
                nf = i + d
                if 1 <= nf <= 18:
                    for lag in [5, 10, 15]:
                        neighbor_temp_lag.append(f'{nf}#风箱温度_lag{lag}')
        
        # 时间窗口统计特征
        stat_cols = [f'{pressure_col}_window{WINDOW}_{stat}' for stat in ['mean', 'std', 'slope']]
        # 大烟道参数滞后
        global_lag_cols = []
        for gcol in GLOBAL_FEATURES:
            for lag in GLOBAL_LAGS:
                global_lag_cols.append(f'{gcol}_lag{lag}')
        
        feature_cols = ([pressure_col] + lag_cols + neighbor_pressure + neighbor_temp +
                        neighbor_temp_lag + stat_cols + GLOBAL_FEATURES + global_lag_cols)
        
        # 尾部增强
        if i >= 11:
            upstream_temp = []
            for u in range(max(1, i-3), i):
                upstream_temp.append(f'{u}#风箱温度')
                for lag in [5, 10, 20]:
                    upstream_temp.append(f'{u}#风箱温度_lag{lag}')
            feature_cols += upstream_temp
            self_temp_lag = [f'{target_col}_lag{t}' for t in range(1, 21)]
            feature_cols += self_temp_lag
        elif i == 5:
            upstream_temp = [f'{u}#风箱温度' for u in [3,4]]
            feature_cols += upstream_temp
            self_temp_lag = [f'{target_col}_lag{t}' for t in range(1, 11)]
            feature_cols += self_temp_lag
        
        # 构造滞后特征（临时对象，仅用于当前风箱）
        # 1. 自身负压滞后
        lag_train = pd.DataFrame({col: train[pressure_col].shift(l) for l, col in zip(range(1, N_LAGS+1), lag_cols)})
        train_temp = pd.concat([train, lag_train], axis=1)
        lag_test = pd.DataFrame({col: test[pressure_col].shift(l) for l, col in zip(range(1, N_LAGS+1), lag_cols)})
        test_temp = pd.concat([test, lag_test], axis=1)
        
        # 2. 相邻风箱温度滞后
        for col in neighbor_temp_lag:
            parts = col.split('_lag')
            src_col = parts[0]
            lag = int(parts[1])
            train_temp[col] = train_temp[src_col].shift(lag)
            test_temp[col] = test_temp[src_col].shift(lag)
        
        # 3. 大烟道参数滞后
        for col in global_lag_cols:
            parts = col.split('_lag')
            src_col = parts[0]
            lag = int(parts[1])
            train_temp[col] = train_temp[src_col].shift(lag)
            test_temp[col] = test_temp[src_col].shift(lag)
        
        # 4. 窗口统计特征
        for stat in ['mean', 'std', 'slope']:
            rolling = train_temp[pressure_col].rolling(window=WINDOW, min_periods=1)
            if stat == 'mean':
                vals = rolling.mean()
            elif stat == 'std':
                vals = rolling.std()
            else:  # slope
                vals = rolling.apply(lambda x: (x.iloc[-1] - x.iloc[0]) / (len(x)-1) if len(x)>1 else 0, raw=False)
            train_temp[f'{pressure_col}_window{WINDOW}_{stat}'] = vals
            rolling_test = test_temp[pressure_col].rolling(window=WINDOW, min_periods=1)
            if stat == 'mean':
                vals_test = rolling_test.mean()
            elif stat == 'std':
                vals_test = rolling_test.std()
            else:
                vals_test = rolling_test.apply(lambda x: (x.iloc[-1] - x.iloc[0]) / (len(x)-1) if len(x)>1 else 0, raw=False)
            test_temp[f'{pressure_col}_window{WINDOW}_{stat}'] = vals_test
        
        # 5. 自身温度滞后（仅对5#和尾部）
        if i == 5 or i >= 11:
            max_lag = 20 if i >= 11 else 10
            for t in range(1, max_lag+1):
                col = f'{target_col}_lag{t}'
                train_temp[col] = train_temp[target_col].shift(t)
                test_temp[col] = test_temp[target_col].shift(t)
        
        # 6. 上游温度滞后（仅对5#和尾部）
        if i == 5:
            for u in [3,4]:
                for lag in [1,5,10]:
                    col = f'{u}#风箱温度_lag{lag}'
                    train_temp[col] = train_temp[f'{u}#风箱温度'].shift(lag)
                    test_temp[col] = test_temp[f'{u}#风箱温度'].shift(lag)
                    feature_cols.append(col)
        elif i >= 11:
            for u in range(max(1, i-3), i):
                for lag in [5,10,20]:
                    col = f'{u}#风箱温度_lag{lag}'
                    if col not in feature_cols:
                        feature_cols.append(col)
                    train_temp[col] = train_temp[f'{u}#风箱温度'].shift(lag)
                    test_temp[col] = test_temp[f'{u}#风箱温度'].shift(lag)
        
        # 删除缺失值
        train_clean = train_temp.dropna(subset=feature_cols).copy()
        test_clean = test_temp.dropna(subset=feature_cols).copy()
        
        X_train = train_clean[feature_cols].values
        y_train = train_clean[target_col].values
        X_test = test_clean[feature_cols].values
        y_test = test_clean[target_col].values
        
        if len(X_test) == 0:
            print(f"警告：{i}#风箱测试集无有效样本，跳过")
            continue
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 选择参数
        if i == 5:
            params = cfg.FAN_PARAMS_5
        # elif i == 1:
        #     params = cfg.FAN_PARAMS_1
        elif i >= 10:
            params = cfg.FAN_PARAMS_TAIL
        else:
            params = cfg.FAN_PARAMS_STABLE
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        models.append(model)
        scalers.append(scaler)

        y_pred = model.predict(X_test)
        y_preds.append(y_pred)
        y_tests.append(y_test)
        # 记录测试集的负压值（第一列）
        test_pressures = X_test[:, 0] 
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        
        print(f"{i}#风箱: MAE={mae:.2f}°C, RMSE={rmse:.2f}°C, R²={r2:.4f}")
    
    print("========== 问题1整体平均性能 ==========")
    print(f"平均MAE: {np.mean(mae_list):.2f} °C")
    print(f"平均RMSE: {np.mean(rmse_list):.2f} °C")
    print(f"平均R²: {np.mean(r2_list):.4f}")
    
    return {
        'models': models,
        'scalers': scalers,
        'mae_list': mae_list,
        'rmse_list': rmse_list,
        'r2_list': r2_list,
        'y_tests': y_tests,
        'y_preds': y_preds,
        'test_pressures': test_pressures
    }


def train_co_models(df, cfg=None):
    """
    训练CO浓度预测模型（分阶段建模）
    
    Parameters
    ----------
    df : pd.DataFrame
        已合并南北侧温度的数据
    cfg : Config, optional
    
    Returns
    -------
    dict: {
        'model_high': LGBMRegressor,
        'model_low': LGBMRegressor,
        'scaler_high': StandardScaler,
        'scaler_low': StandardScaler,
        'features': list,
        'train_high': pd.DataFrame,
        'train_low': pd.DataFrame
    }
    """
    if cfg is None:
        cfg = Config()
    
    df_co = df.copy()
    target = '烧结大烟道外排CO浓度'
    
    # ----- 特征构造 -----
    # CO自身滞后
    for lag in cfg.CO_AR_LAGS:
        df_co[f'{target}_lag{lag}'] = df_co[target].shift(lag)
    # 差分
    df_co['CO_diff1'] = df_co[target].diff(1)
    df_co['CO_diff2'] = df_co[target].diff(2)
    # 滚动统计
    for w in cfg.CO_ROLLING_WINDOWS:
        df_co[f'CO_mean_{w}'] = df_co[target].rolling(w).mean()
        df_co[f'CO_std_{w}'] = df_co[target].rolling(w).std()
    # 关键风箱温度
    for fan in cfg.CO_TEMP_FANS:
        df_co[f'{fan}温度'] = df_co[f'{fan}#风箱温度']
        df_co[f'{fan}温度_diff'] = df_co[f'{fan}温度'].diff(5)
    # 大烟道和机速
    df_co['烟道1'] = df_co['1#大烟道温度']
    df_co['烟道2'] = df_co['2#大烟道温度']
    df_co['机速'] = df_co['烧结机机速L1设定']
    
    df_co = df_co.dropna().reset_index(drop=True)
    print("========== 问题2：CO浓度预测 ==========")
    print(f"特征构造后数据量: {len(df_co)}")
    
    # 划分训练集
    train_size = int(cfg.TRAIN_SPLIT * len(df_co))
    train_data = df_co.iloc[:train_size].copy()
    train_data['CO_ma30'] = train_data[target].rolling(cfg.CO_WINDOW_STATS).mean()
    mean_values = train_data['CO_ma30'].dropna().values
    diff_means = np.diff(mean_values)
    split_idx = np.argmin(diff_means) + cfg.CO_WINDOW_STATS
    print(f"训练集内工况切换点索引: {split_idx}")
    
    train_high = train_data.iloc[:split_idx].copy()
    train_low = train_data.iloc[split_idx:].copy()
    print(f"高CO阶段样本数: {len(train_high)}, 低CO阶段样本数: {len(train_low)}")
    
    features = [c for c in df_co.columns if c not in [target, '序列']]
    
    scaler_high = StandardScaler()
    scaler_low = StandardScaler()
    X_high = scaler_high.fit_transform(train_high[features].values)
    y_high = train_high[target].values
    X_low = scaler_low.fit_transform(train_low[features].values)
    y_low = train_low[target].values
    
    model_high = lgb.LGBMRegressor(**cfg.CO_MODEL_PARAMS)
    model_low = lgb.LGBMRegressor(**cfg.CO_MODEL_PARAMS)
    model_high.fit(X_high, y_high)
    model_low.fit(X_low, y_low)
    
    # 测试集预测（仅用于评估，不返回预测值）
    test_data = df_co.iloc[train_size:].copy()
    test_data['CO_ma30'] = test_data[target].rolling(cfg.CO_WINDOW_STATS, min_periods=1).mean()
    threshold = cfg.CO_THRESHOLD
    X_test = test_data[features].values
    y_test = test_data[target].values
    y_pred = np.zeros_like(y_test)
    
    for i in range(len(test_data)):
        if i < cfg.CO_WINDOW_STATS:
            scaler, model = scaler_high, model_high
        else:
            if test_data['CO_ma30'].iloc[i] > threshold:
                scaler, model = scaler_high, model_high
            else:
                scaler, model = scaler_low, model_low
        X_scaled = scaler.transform(X_test[i:i+1])
        y_pred[i] = model.predict(X_scaled)[0]
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("========== 问题2：CO浓度预测 ==========")
    print(f"MAE: {mae:.2f} mg/m³")
    print(f"RMSE: {rmse:.2f} mg/m³")
    print(f"R²: {r2:.4f}")
    
    return {
        'model_high': model_high,
        'model_low': model_low,
        'scaler_high': scaler_high,
        'scaler_low': scaler_low,
        'features': features,
        'train_high': train_high,
        'train_low': train_low,
        'y_test': y_test,
        'y_pred': y_pred,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'split_idx': split_idx,
        'train_high': train_high,
        'train_low': train_low
    }