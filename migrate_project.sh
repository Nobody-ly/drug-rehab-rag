#!/bin/bash

OLD_DIR="$HOME/Ly/rag_qa"
NEW_DIR="$HOME/Ly/drug-rehab-rag"

echo "📦 开始迁移核心文件..."
echo "  源目录: $OLD_DIR"
echo "  目标目录: $NEW_DIR"
echo ""

# ============================================
# 1. 核心代码
# ============================================
echo "  → 迁移核心代码..."

cp "$OLD_DIR/enhanced_rag_system.py" "$NEW_DIR/core/rag_system.py" 2>/dev/null && echo "    ✅ rag_system.py" || echo "    ⚠️  enhanced_rag_system.py 未找到"
cp "$OLD_DIR/semantic_quality_eval.py" "$NEW_DIR/core/" 2>/dev/null && echo "    ✅ semantic_quality_eval.py" || echo "    ⚠️  semantic_quality_eval.py 未找到"
cp "$OLD_DIR/quality_evaluation.py" "$NEW_DIR/core/" 2>/dev/null && echo "    ✅ quality_evaluation.py" || echo "    ⚠️  quality_evaluation.py 未找到"
cp "$OLD_DIR/build_jieduo_vectordb.py" "$NEW_DIR/core/build_vectordb.py" 2>/dev/null && echo "    ✅ build_vectordb.py" || echo "    ⚠️  build_jieduo_vectordb.py 未找到"
cp "$OLD_DIR/document_loader.py" "$NEW_DIR/core/" 2>/dev/null && echo "    ✅ document_loader.py" || echo "    ⚠️  document_loader.py 未找到"

echo ""

# ============================================
# 2. 压测代码
# ============================================
echo "  → 迁移压测代码..."

cp "$OLD_DIR/full_benchmark_jieduo.py" "$NEW_DIR/benchmark/full_benchmark.py" 2>/dev/null && echo "    ✅ full_benchmark.py" || echo "    ⚠️  full_benchmark_jieduo.py 未找到"
cp "$OLD_DIR/vllm_optimization_v4.py" "$NEW_DIR/benchmark/vllm_optimization.py" 2>/dev/null && echo "    ✅ vllm_optimization.py" || echo "    ⚠️  vllm_optimization_v4.py 未找到"
cp "$OLD_DIR/load_test.py" "$NEW_DIR/benchmark/" 2>/dev/null && echo "    ✅ load_test.py" || echo "    ⚠️  load_test.py 未找到"
cp "$OLD_DIR/async_load_test_ttft.py" "$NEW_DIR/benchmark/measure_ttft.py" 2>/dev/null && echo "    ✅ measure_ttft.py" || echo "    ⚠️  async_load_test_ttft.py 未找到"

echo ""

# ============================================
# 3. 数据文件
# ============================================
echo "  → 迁移数据文件..."

# 问题集
if [ -d "$OLD_DIR/data" ]; then
    cp "$OLD_DIR/data"/*.json "$NEW_DIR/data/questions/" 2>/dev/null && echo "    ✅ 问题集已复制" || echo "    ⚠️  问题集未找到"
else
    echo "    ⚠️  data目录不存在"
fi

# 知识库文档
if [ -d "$OLD_DIR/data/jieduo_documents" ]; then
    cp -r "$OLD_DIR/data/jieduo_documents" "$NEW_DIR/data/documents/" && echo "    ✅ 知识库文档"
else
    echo "    ⚠️  知识库文档目录未找到"
fi

echo ""

# ============================================
# 4. 测试结果
# ============================================
echo "  → 迁移测试结果..."

cp "$OLD_DIR/jieduo_benchmark_report_20260128_115652.json" "$NEW_DIR/results/benchmark/latest_report.json" 2>/dev/null && echo "    ✅ 压测报告" || echo "    ⚠️  压测报告未找到"
cp "$OLD_DIR/jieduo_benchmark_results_20260128_115652.json" "$NEW_DIR/results/benchmark/latest_results.json" 2>/dev/null && echo "    ✅ 压测结果" || echo "    ⚠️  压测结果未找到"
cp "$OLD_DIR/vllm_optimization_v4_report_20260127_160242.json" "$NEW_DIR/results/optimization/vllm_optimization.json" 2>/dev/null && echo "    ✅ vLLM优化" || echo "    ⚠️  vLLM优化未找到"
cp "$OLD_DIR/quality_evaluation_report_20260128_104455.json" "$NEW_DIR/results/evaluation/quality_report.json" 2>/dev/null && echo "    ✅ 质量评估" || echo "    ⚠️  质量评估未找到"

echo ""

# ============================================
# 5. 文档
# ============================================
echo "  → 迁移文档..."

cp "$OLD_DIR/FINAL_BENCHMARK_REPORT.md" "$NEW_DIR/docs/" 2>/dev/null && echo "    ✅ 完整报告" || echo "    ⚠️  FINAL_BENCHMARK_REPORT.md 未找到"
cp "$OLD_DIR/performance_baseline.md" "$NEW_DIR/docs/" 2>/dev/null && echo "    ✅ 性能基线" || echo "    ⚠️  performance_baseline.md 未找到"

echo ""

# ============================================
# 6. 配置文件
# ============================================
echo "  → 迁移配置文件..."

cp "$OLD_DIR/pyproject.toml" "$NEW_DIR/" 2>/dev/null && echo "    ✅ pyproject.toml" || echo "    ⚠️  pyproject.toml 未找到"

echo ""

# ============================================
# 汇总
# ============================================
echo "✅ 迁移完成！"
echo ""
echo "📊 新项目统计："
echo "  核心代码:   $(find "$NEW_DIR/core" -name "*.py" 2>/dev/null | wc -l) 个文件"
echo "  压测代码:   $(find "$NEW_DIR/benchmark" -name "*.py" 2>/dev/null | wc -l) 个文件"
echo "  数据文件:   $(find "$NEW_DIR/data" -type f 2>/dev/null | wc -l) 个文件"
echo "  测试结果:   $(find "$NEW_DIR/results" -name "*.json" 2>/dev/null | wc -l) 个文件"
echo "  文档:       $(find "$NEW_DIR/docs" -name "*.md" 2>/dev/null | wc -l) 个文件"
echo ""
echo "📁 新项目目录: $NEW_DIR"
echo ""
echo "🔍 查看文件列表:"
echo "   ls -lh $NEW_DIR/core/"
echo "   ls -lh $NEW_DIR/benchmark/"
