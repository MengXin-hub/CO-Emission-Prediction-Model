"""
optimization.py - 优化模块
"""
import numpy as np
from scipy.optimize import differential_evolution
from config import Config

def optimize_pressure(model, scaler, features, train_high, df, cfg=None):
    """
    基于高CO阶段模型，使用差分进化算法优化风箱负压
    
    Parameters
    ----------
    model : lgb.Booster
        训练好的高CO阶段LightGBM模型
    scaler : StandardScaler
        训练时使用的标准化器
    features : list
        特征列名列表（顺序需与训练时一致）
    train_high : pd.DataFrame
        高CO阶段训练集（用于获取固定均值）
    df : pd.DataFrame
        原始数据（用于获取负压边界）
    cfg : Config, optional
    
    Returns
    -------
    dict: {
        'optimal_pressures': np.ndarray,  # 最优负压（长度18）
        'optimal_co': float,              # 最优CO浓度预测值
        'baseline_pressures': list,       # 基准负压（均值）
        'baseline_co': float,             # 基准CO浓度
        'lowering': float                 # CO降低量
    }
    """
    if cfg is None:
        cfg = Config()
    
    # 1. 确定负压边界（各风箱最小/最大值）
    bounds = []
    for i in range(1, 19):
        col = f'{i}#风箱负压'
        min_val = df[col].min()
        max_val = df[col].max()
        bounds.append((min_val, max_val))
        print(f"{i}#风箱负压范围: [{min_val:.3f}, {max_val:.3f}]")
    
    # 2. 固定其他特征为高CO阶段训练集均值
    fixed_means = train_high[features].mean()
    print("使用高CO阶段训练集均值作为固定工况")
    
    # 3. 目标函数
    def objective(x):
        # x: 长度为18的数组，对应1~18#风箱负压
        feat_dict = {}
        for i in range(1, 19):
            feat_dict[f'{i}#风箱负压'] = x[i-1]
        # 其他特征使用固定均值
        for col in features:
            if col not in feat_dict:
                feat_dict[col] = fixed_means[col]
        # 按特征顺序构造特征向量
        X = np.array([feat_dict[col] for col in features]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)[0]
    
    # 4. 差分进化优化
    print("开始优化（单进程模式）...")
    result = differential_evolution(
        objective, bounds,
        maxiter=cfg.OPT_MAXITER,
        popsize=cfg.OPT_POPSIZE,
        seed=cfg.OPT_SEED,
        workers=cfg.OPT_WORKERS,
        updating='deferred'
    )
    
    optimal_pressures = result.x
    optimal_co = result.fun
    print("优化完成！")
    print(f"最优CO浓度预测值 = {optimal_co:.2f} mg/m³")
    print("各风箱最优负压（kPa）：")
    for i, p in enumerate(optimal_pressures, 1):
        print(f"{i}#风箱: {p:.3f}")
    
    # 5. 基准工况（各风箱负压取均值）
    baseline_pressures = [df[f'{i}#风箱负压'].mean() for i in range(1, 19)]
    baseline_co = objective(baseline_pressures)
    print(f"基准工况（均值）CO浓度 = {baseline_co:.2f} mg/m³")
    print(f"优化后CO浓度降低量 = {baseline_co - optimal_co:.2f} mg/m³")
    
    # 6. 调控建议
    print("========== 优化调控建议（与基准值对比） ==========")
    for i in range(1, 19):
        diff = optimal_pressures[i-1] - baseline_pressures[i-1]
        if diff > 0:
            print(f"{i}#风箱负压应升高 {diff:.3f} kPa")
        elif diff < 0:
            print(f"{i}#风箱负压应降低 {abs(diff):.3f} kPa")
        else:
            print(f"{i}#风箱负压保持不变")
    
    return {
        'optimal_pressures': optimal_pressures,
        'optimal_co': optimal_co,
        'baseline_pressures': baseline_pressures,
        'baseline_co': baseline_co,
        'lowering': baseline_co - optimal_co
    }