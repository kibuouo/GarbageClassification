# 课程成果

将需要随项目交付的最终图表、指标文件和报告放入 `results/final/` 并提交到 Git。

例如：训练曲线、混淆矩阵、最终测试指标、误分类分析、课程报告。其他临时运行文件默认忽略。

`train.py` 将 A/B/C 的日志、配置、检查点和逐轮指标写入 `runs/<实验>/<run-id>/`。`evaluate.py --experiments A B C` 生成各运行的测试指标，并汇总为 `final/test_metrics.json` 和 `final/comparison.csv`；`plot_results.py` 生成 `final/training_curves.png`、`final/confusion_matrix.png`。只有完成真实训练和评估后才会生成正式成果文件。使用方式见项目根目录 README。

`final/validation_baseline.json`、`final/finetuning_history.json` 属于先前的探索性验证，不是 A/B/C 结果。原始随附权重缺少训练来源记录，不能将这些验证指标写成独立测试准确率。

`smoke/` 仅保存小样本开发检查，图表有 SMOKE TEST 标记，默认排除在正式汇总之外。`legacy/train_before_experiments.py` 是重构前的训练脚本备份。
