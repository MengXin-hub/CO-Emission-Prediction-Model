"""
visualization.py - 可视化模块
"""
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import pandas as pd

# 设置全局中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 问题一
def plot_fan_pressure_temp_curves(df, fan_results, test_indices, fan_indices=None, save_path=None, csv_dir=None):
    """
    每个风箱单独绘制负压-温度曲线（按负压排序后连线）
    
    Parameters
    ----------
    df : pd.DataFrame
        原始数据（包含所有列）
    fan_results : dict
        需包含 'y_tests', 'y_preds'
    test_indices : array-like
        测试集在 df 中的索引（长度必须与 y_tests[0] 一致）
    fan_indices : list, optional
    save_path : str, optional
    csv_dir : str, optional
    """
    if fan_indices is None:
        fan_indices = list(range(1, 19))
    
    # 提取与预测值长度匹配的测试集数据
    test_df = df.iloc[test_indices].copy()
    
    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, fan in enumerate(fan_indices):
        ax = axes[idx]
        pressure_col = f'{fan}#风箱负压'
        temp_col = f'{fan}#风箱温度'
        
        # 真实温度
        true_temp = test_df[temp_col].values
        # 预测温度
        pred_temp = fan_results['y_preds'][fan-1]
        # 负压值
        pressure = test_df[pressure_col].values
        
        # 按负压排序
        sort_idx = np.argsort(pressure)
        pressure_sorted = pressure[sort_idx]
        true_sorted = true_temp[sort_idx]
        pred_sorted = pred_temp[sort_idx]
        
        ax.plot(pressure_sorted, true_sorted, color='gold', linewidth=2, label='实际值')
        ax.plot(pressure_sorted, pred_sorted, color='red', linewidth=2, label='预测值')
        ax.set_xlabel('负压 (kPa)')
        ax.set_ylabel('温度 (°C)')
        ax.set_title(f'{fan}#风箱')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if csv_dir:
            data = pd.DataFrame({
                    '负压_kPa': pressure_sorted,
                    '实际温度_°C': true_sorted,
                    '预测温度_°C': pred_sorted
                })
            filename = f'fan_{fan}_pressure_temp.csv'
            save_to_csv(data, filename, csv_dir)
    
    # 隐藏多余子图
    for j in range(len(fan_indices), len(axes)):
        axes[j].set_visible(False)
    
    
    
    plt.suptitle('各风箱负压-温度曲线（实际值 VS 预测值）', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_overall_prediction_curve(fan_results, test_indices, save_path=None, csv_dir=None):
    """
    绘制总体预测曲线：所有风箱实际平均温度 vs 预测平均温度
    """
    n_samples = len(test_indices)
    n_fans = len(fan_results['y_tests'])
    
    true_avg = np.zeros(n_samples)
    pred_avg = np.zeros(n_samples)
    for i in range(n_fans):
        true_avg += fan_results['y_tests'][i]
        pred_avg += fan_results['y_preds'][i]
    true_avg /= n_fans
    pred_avg /= n_fans

    if csv_dir:
        data = pd.DataFrame({
            '样本序号': range(1, len(true_avg)+1),
            '实际平均温度_°C': true_avg,
            '预测平均温度_°C': pred_avg
        })
        save_to_csv(data, 'overall_prediction.csv', csv_dir)
    
    plt.figure(figsize=(14, 6))
    plt.plot(true_avg, color='gold', linewidth=2, label='实际平均温度')
    plt.plot(pred_avg, color='red', linewidth=2, label='预测平均温度')
    plt.xlabel('测试样本序号')
    plt.ylabel('温度 (°C)')
    plt.title('总体预测效果（所有风箱平均温度）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 问题二
def plot_co_influence_curves(df, co_results, save_path=None, csv_dir=None):
    """
    展示各变量对CO浓度的影响曲线（散点图+趋势线）
    选取代表性特征：部分风箱温度、负压、大烟道参数
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import linregress
    
    # 选取特征列
    features_to_plot = [
        ('13#风箱温度', '温度(°C)'), ('18#风箱温度', '温度(°C)'),
        ('1#大烟道温度', '温度(°C)'), ('2#大烟道温度', '温度(°C)'),
        ('1#风箱负压', '负压(kPa)'), ('9#风箱负压', '负压(kPa)'), ('18#风箱负压', '负压(kPa)'),
        ('1#大烟道负压', '负压(kPa)'), ('2#大烟道负压', '负压(kPa)')
    ]
    n_cols = 3
    n_rows = (len(features_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    target = '烧结大烟道外排CO浓度'
    for idx, (feat, unit) in enumerate(features_to_plot):
        ax = axes[idx]
        ax.scatter(df[feat], df[target], s=5, alpha=0.3, c='blue')
        # 添加线性趋势线
        x = df[feat].values
        y = df[target].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 1:
            slope, intercept, r_value, _, _ = linregress(x[mask], y[mask])
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r-', label=f'趋势线 (r={r_value:.2f})')
        ax.set_xlabel(f'{feat} ({unit})')
        ax.set_ylabel('CO浓度 (mg/$m^3$)')
        ax.set_title(f'{feat} vs CO')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if csv_dir:
            data = pd.DataFrame({
                f'{feat}': x,
                'CO浓度_mg_m3': y
            })
            # 去除缺失值
            data = data.dropna()
            # 文件名含特殊字符需转义，用下划线替换
            safe_name = feat.replace('#', '_').replace(' ', '_')
            filename = f'influence_{safe_name}.csv'
            save_to_csv(data, filename, csv_dir)


    for j in range(len(features_to_plot), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pressure_co_response(model, scaler, features, train_high, df, fan_idx, pressure_range=None, save_path=None, csv_dir=None):
    """
    绘制特定风箱负压对CO浓度的影响曲线（固定其他特征为高CO阶段均值）
    """
    if pressure_range is None:
        pressure_col = f'{fan_idx}#风箱负压'
        pressure_range = (df[pressure_col].min(), df[pressure_col].max())
    pressures = np.linspace(pressure_range[0], pressure_range[1], 100)
    # 固定其他特征为高CO阶段训练集均值
    fixed_means = train_high[features].mean()
    co_vals = []
    for p in pressures:
        feat_dict = fixed_means.to_dict()
        feat_dict[f'{fan_idx}#风箱负压'] = p
        # 按特征顺序构造向量
        X = np.array([feat_dict[col] for col in features]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        co_vals.append(model.predict(X_scaled)[0])
    
    if csv_dir:
        data = pd.DataFrame({
            f'{fan_idx}#风箱负压_kPa': pressures,
            '预测CO浓度_mg_m3': co_vals
        })
        filename = f'pressure_co_response_fan{fan_idx}.csv'
        save_to_csv(data, filename, csv_dir)

    plt.figure(figsize=(8,5))
    plt.plot(pressures, co_vals, 'b-', linewidth=2)
    plt.xlabel(f'{fan_idx}#风箱负压 (kPa)')
    plt.ylabel('预测CO浓度 (mg/$m^3$)')
    plt.title(f'{fan_idx}#风箱负压对CO浓度的影响（固定其他工况）')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_co_prediction_curve(y_true, y_pred, split_point=None, save_path=None, csv_dir=None):
    """
    绘制CO浓度预测值与真实值对比曲线
    """
    if csv_dir:
        data = pd.DataFrame({
            '样本序号': range(1, len(y_true)+1),
            '实际CO浓度_mg_m3': y_true,
            '预测CO浓度_mg_m3': y_pred
        })
        save_to_csv(data, 'co_prediction.csv', csv_dir)

    plt.figure(figsize=(14, 6))
    plt.plot(y_true, color='gold', linewidth=1.5, label='实际CO浓度')
    plt.plot(y_pred, color='red', linewidth=1.5, label='预测CO浓度')
    if split_point is not None:  # 绘制分割线
        plt.axvline(x=split_point, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('测试样本序号')
    plt.ylabel('CO浓度 (mg/$m^3$)')
    plt.title('CO浓度预测效果（实际值 VS 预测值）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 问题三
def plot_co_reduction_comparison(baseline_co, optimal_co, save_path=None, csv_dir=None):
    """
    绘制基准与优化后CO浓度对比图（棒棒糖图）
    """
    import matplotlib.pyplot as plt
    labels = ['基准工况\n(均值负压)', '优化工况\n(最优负压)']
    values = [baseline_co, optimal_co]
    colors = ['#1f77b4', '#ff7f0e']
    reduction = baseline_co - optimal_co
    percent = (reduction / baseline_co) * 100

    if csv_dir:
        data = pd.DataFrame({
            '工况': ['基准工况', '优化工况'],
            'CO浓度_mg_m3': [baseline_co, optimal_co],
            '降低量_mg_m3': [0, reduction]
        })
        save_to_csv(data, 'co_reduction_comparison.csv', csv_dir)

    plt.figure(figsize=(6, 5))
    # 棒棒糖图：竖线 + 圆点
    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        plt.plot([i, i], [0, val], color=color, linewidth=3, marker='o', markersize=12)
        plt.text(i, val + 15, f'{val:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xticks([0, 1], labels, fontsize=12)
    plt.ylabel('CO浓度 (mg/$m^3$)', fontsize=12)
    plt.title(f'优化效果：CO浓度降低 {reduction:.1f} mg/$m^3$ ({percent:.1f}%)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.xlim(-0.5, 1.5) 
    plt.show()


def plot_co_reduction_waterfall(baseline_co, optimal_co, save_path=None, csv_dir=None):
    """
    绘制CO浓度变化的瀑布图（从基准到优化，展示变化量）
    """
    import matplotlib.pyplot as plt
    reduction = baseline_co - optimal_co
    steps = ['基准工况', '优化调整', '优化工况']
    values = [baseline_co, reduction, optimal_co]
    colors = ['#1f77b4', '#2ca02c' if reduction>0 else '#d62728', '#ff7f0e']

    if csv_dir:
        data = pd.DataFrame({
            '阶段': steps,
            'CO浓度_mg_m3': values
        })
        save_to_csv(data, 'co_reduction_waterfall.csv', csv_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    # 瀑布图通常用柱子表示，此处用简单柱状图
    bars = ax.bar(steps, values, color=colors, width=0.5)
    # 标注数值
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10, f'{val:.0f}',
                ha='center', va='bottom', fontweight='bold')
    # 添加箭头表示降低方向
    ax.annotate('', xy=(1, optimal_co), xytext=(1, baseline_co),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(1, (baseline_co+optimal_co)/2, f'降低 {reduction:.1f} mg/$m^3$',
            ha='center', va='center', fontsize=10, color='red', backgroundcolor='white')
    ax.set_ylabel('CO浓度 (mg/$m^3$)')
    ax.set_title('CO浓度优化瀑布图')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 保存数据csv
def save_to_csv(data, filename, csv_dir):
    """将数据保存为 CSV 文件，若目录不存在则创建"""
    if csv_dir is None:
        return
    os.makedirs(csv_dir, exist_ok=True)
    filepath = os.path.join(csv_dir, filename)
    data.to_csv(filepath, index=False, encoding='utf-8-sig')
    # print(f"数据已保存至: {filepath}")