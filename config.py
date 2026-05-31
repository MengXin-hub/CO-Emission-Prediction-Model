"""
config.py - 配置文件
"""
import os

class Config:
    # 数据文件路径
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "data.xlsx")

    LOG_PATH = os.path.join(BASE_DIR, "log.txt")

    # CSV 输出根目录
    CSV_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'csv_output')
    
    # 各主题对应的 CSV 文件名（相对路径）
    CSV_FILES = {
        'fan_pressure_temp': 'fan_{fan}_pressure_temp.csv',        # 每个风箱一个文件
        'overall_prediction': 'overall_prediction.csv',
        'co_influence': 'co_influence.csv',
        'pressure_co_response': 'pressure_co_response_fan{fan}.csv',
        'co_prediction': 'co_prediction.csv',
        'co_reduction': 'co_reduction.csv',
        'pressure_optimization': 'pressure_optimization.csv'
    }

    # 全局特征列名
    GLOBAL_FEATURES = [
        '烧结机机速L1设定',
        '1#大烟道负压', '2#大烟道负压',
        '1#大烟道温度', '2#大烟道温度'
    ]
    # ---------- 问题一：风箱温度预测超参数 ----------
    FAN_N_LAGS = 60  # 自身负压滞后步数（120秒）
    FAN_NEIGHBORS = 2  # 相邻风箱范围（前后各2个）
    FAN_WINDOW = 10  # 滑动窗口大小
    FAN_GLOBAL_LAGS = [10, 20, 30]  # 大烟道参数滞后步数

    # 不同风箱段的LightGBM参数
    FAN_PARAMS_STABLE = {  # 稳定区（1-10风箱除5风箱）
        'n_estimators': 500, 'max_depth': 10, 'learning_rate': 0.05,
        'num_leaves': 40, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_lambda': 0.1, 'reg_alpha': 0.1, 'min_child_samples': 5,
        'random_state': 42, 'n_jobs': -1, 'verbosity': -1
    }
    FAN_PARAMS_5 = {  # 5风箱特殊参数
        'n_estimators': 800, 'max_depth': 14, 'learning_rate': 0.03,
        'num_leaves': 70, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_lambda': 1.0, 'reg_alpha': 0.5, 'min_child_samples': 10,
        'random_state': 42, 'n_jobs': -1, 'verbosity': -1
    }
    FAN_PARAMS_TAIL = {  # 尾部风箱（11-18）
        'n_estimators': 700, 'max_depth': 12, 'learning_rate': 0.04,
        'num_leaves': 60, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_lambda': 0.5, 'reg_alpha': 0.2, 'min_child_samples': 5,
        'random_state': 42, 'n_jobs': -1, 'verbosity': -1
    }

    # ---------- 问题二：CO浓度预测超参数 ----------
    CO_AR_LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30,
                  40, 50, 60, 80, 100, 120, 150, 180, 200]   # CO自身滞后
    CO_ROLLING_WINDOWS = [10, 20]        # 滚动窗口大小（用于均值、标准差）
    CO_WINDOW_STATS = 20                 # 用于工况切换的滑动窗口
    CO_TEMP_FANS = [13, 14, 18]          # 关键风箱（温度特征）
    CO_THRESHOLD = 2800                  # 高/低CO阶段阈值,3500
    CO_MODEL_PARAMS = {
        'n_estimators': 800, 'max_depth': 12, 'learning_rate': 0.05,
        'num_leaves': 60, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_lambda': 0.01, 'reg_alpha': 0.01, 'min_child_samples': 5,
        'random_state': 42, 'n_jobs': -1, 'verbosity': -1
    }
    
    # ---------- 问题三：优化超参数 ----------
    OPT_MAXITER = 100       # 差分进化最大迭代次数50
    OPT_POPSIZE = 50       # 种群大小20
    OPT_SEED = 42          # 随机种子42
    OPT_WORKERS = 1        # 进程数（1避免Windows多进程错误）
    
    # ---------- 其他 ----------
    TRAIN_SPLIT = 0.7      # 训练集比例（按时间顺序前70%）
