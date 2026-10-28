# 课程成果

将需要随项目交付的最终图表、指标文件和报告放入 `results/final/` 并提交到 Git。

本次三组独立测试结果：A 82.54%、B 85.45%、C 92.86%，使用相同的 378 张测试图片。`final/accuracy_comparison.png` 和 `.svg` 可用于演示，`comparison_notes.md` 说明方法和限制；`training_logs/<A|B|C>/` 保存对应的配置、元数据、训练日志、逐轮历史和完成状态，不含模型及优化器检查点。`training_logs/manifest.json` 记录归档文件的 SHA-256。

仓库中的图表可直接查看。重新执行 `plot_results.py` 需要完整的本地 `results/runs/` 目录；新克隆仓库可先按照根目录 README 训练并评估，生成新的运行记录和图表。

例如：训练曲线、混淆矩阵、最终测试指标、误分类分析、课程报告。其他临时运行文件默认忽略。

`train.py` 将 A/B/C 的日志、配置、检查点和逐轮指标写入 `runs/<实验>/<run-id>/`。`evaluate.py --experiments A B C` 生成各运行的测试指标，并汇总为 `final/test_metrics.json` 和 `final/comparison.csv`；`plot_results.py` 生成 `final/training_curves.png`、`final/confusion_matrix.png`。只有完成真实训练和评估后才会生成正式成果文件。使用方式见项目根目录 README。

`final/validation_baseline.json`、`final/finetuning_history.json` 属于先前的探索性验证，不是 A/B/C 结果。原始随附权重缺少训练来源记录，不能将这些验证指标写成独立测试准确率。

`smoke/` 仅保存小样本开发检查，图表有 SMOKE TEST 标记，默认排除在正式汇总之外。`legacy/train_before_experiments.py` 是重构前的训练脚本备份。
