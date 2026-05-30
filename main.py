"""
main.py - 主程序入口
"""
from config import Config
from data_loader import load_and_preprocess
from models import train_fan_models, train_co_models
from optimization import optimize_pressure
import warnings
warnings.filterwarnings('ignore')
from visualization import *
from logger import setup_logger, log_model_results
import logging



def main():
    # 1. 加载配置
    cfg = Config()
    csv_dir = cfg.CSV_OUTPUT_DIR

    # 初始化日志
    logger = setup_logger(log_file=cfg.LOG_PATH, level=logging.INFO)
    print("配置加载成功!")
    
    # 2. 加载并预处理数据
    df = load_and_preprocess(cfg)
    print(f"数据加载完成，共 {len(df)} 行")
    
    # 3. 问题一：风箱温度预测
    print("\n" + "="*60)
    print("问题一：风箱温度预测")
    print("="*60)
    fan_results = train_fan_models(df, cfg)
    # 获取测试集实际有效样本数（预测值长度）
    n_valid_test = len(fan_results['y_tests'][0])
    # 原始测试集索引（前 train_size 之后的所有索引）
    train_size = int(cfg.TRAIN_SPLIT * len(df))
    full_test_indices = df.index[train_size:].to_numpy()
    # 截取最后 n_valid_test 个索引（因为 dropna 删除了前面的缺失行）
    valid_test_indices = full_test_indices[-n_valid_test:]
    # 绘制每个风箱的负压-温度曲线
    plot_fan_pressure_temp_curves(df, fan_results, valid_test_indices, save_path=None, csv_dir=csv_dir)
    # 绘制总体预测曲线（同样传入有效索引）
    plot_overall_prediction_curve(fan_results, valid_test_indices, save_path=None, csv_dir=csv_dir)

    logger.info("========== 问题一：风箱温度预测 ==========")
    # 记录每个风箱的详细指标
    for i in range(1, 19):
        logger.info(f"{i}#风箱: MAE={fan_results['mae_list'][i-1]:.2f}°C, RMSE={fan_results['rmse_list'][i-1]:.2f}°C, R²={fan_results['r2_list'][i-1]:.4f}")
    # 记录整体平均性能
    avg_metrics = {
        '平均MAE': f"{np.mean(fan_results['mae_list']):.2f} °C",
        '平均RMSE': f"{np.mean(fan_results['rmse_list']):.2f} °C",
        '平均R²': f"{np.mean(fan_results['r2_list']):.4f}"
    }
    log_model_results(logger, '问题一整体性能', avg_metrics)
    
    # 4. 问题二：CO浓度预测（分阶段建模）
    print("\n" + "="*60)
    print("问题二：CO浓度预测")
    print("="*60)
    co_results = train_co_models(df, cfg)
    # 1. 多变量影响曲线（使用原始数据）
    plot_co_influence_curves(df, co_results, save_path=None, csv_dir=csv_dir)
    # 2. 重点风箱负压对CO的影响（例如5#、12#、18#风箱）
    for fan in [5, 12, 18]:
        plot_pressure_co_response(co_results['model_high'], co_results['scaler_high'], 
                                co_results['features'], co_results['train_high'], 
                                df, fan, save_path=None, csv_dir=csv_dir)
    # 3. CO预测对比曲线
    plot_co_prediction_curve(co_results['y_test'], co_results['y_pred'], split_point=125, save_path=None, csv_dir=csv_dir)
    # 记录评估指标
    co_metrics = {
        'MAE': f"{co_results['mae']:.2f} mg/$m^3$",
        'RMSE': f"{co_results['rmse']:.2f} mg/$m^3$",
        'R²': f"{co_results['r2']:.4f}"
    }
    log_model_results(logger, '问题二CO浓度预测', co_metrics)
    # 记录分阶段信息
    logger.info(f"分阶段建模：工况切换点索引 = {co_results['split_idx']}")
    logger.info(f"高CO阶段样本数 = {len(co_results['train_high'])}, 低CO阶段样本数 = {len(co_results['train_low'])}")
   
    # 5. 问题三：风箱负压优化（基于高CO阶段模型）
    print("\n" + "="*60)
    print("问题三：风箱负压优化")
    print("="*60)
    opt_results = optimize_pressure(
        model=co_results['model_high'],
        scaler=co_results['scaler_high'],
        features=co_results['features'],
        train_high=co_results['train_high'],
        df=df,
        cfg=cfg
    )
    plot_co_reduction_comparison(opt_results['baseline_co'], opt_results['optimal_co'], save_path=None, csv_dir=csv_dir)
    plot_co_reduction_waterfall(opt_results['baseline_co'], opt_results['optimal_co'], save_path=None, csv_dir=csv_dir)

    logger.info("========== 问题三：风箱负压优化 ==========")
    logger.info(f"基准工况CO浓度: {opt_results['baseline_co']:.2f} mg/$m^3$")
    logger.info(f"优化后CO浓度: {opt_results['optimal_co']:.2f} mg/$m^3$")
    logger.info(f"CO浓度降低量: {opt_results['lowering']:.2f} mg/$m^3$")
    logger.info("各风箱负压调整详情：")
    for i in range(1, 19):
        logger.info(f"{i}#风箱: 基准={opt_results['baseline_pressures'][i-1]:.3f} kPa, 优化={opt_results['optimal_pressures'][i-1]:.3f} kPa")

    logger.info("========== 优化调控建议（与基准值对比） ==========")
    for i in range(1, 19):
        diff = opt_results['optimal_pressures'][i-1] - opt_results['baseline_pressures'][i-1]
        if diff > 0:
            logger.info(f"{i}#风箱负压应升高 {diff:.3f} kPa")
        elif diff < 0:
            logger.info(f"{i}#风箱负压应降低 {abs(diff):.3f} kPa")
        else:
            logger.info(f"{i}#风箱负压保持不变")
    
    print("\n全部任务完成！")

if __name__ == '__main__':
    main()
    