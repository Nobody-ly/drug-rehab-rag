import asyncio
import time
import statistics
import json
from datetime import datetime
from enhanced_rag_system import EnhancedRAGSystem

# 戒毒相关测试问题（10个）
TEST_QUERIES = [
    "什么是强制隔离戒毒？",
    "社区戒毒的适用条件是什么？",
    "戒毒人员有哪些权利和保障？",
    "戒毒期限一般是多久？",
    "如何申请自愿戒毒？",
    "戒毒场所应当提供哪些服务？",
    "禁毒法的主要内容是什么？",
    "吸毒成瘾的认定标准是什么？",
    "戒毒康复人员就业有什么政策？",
    "家属可以探视戒毒人员吗？",
]

CONCURRENCY = 10   # 并发数
TOTAL_REQUESTS = 30  # 总请求数

class RAGBenchmark:
    def __init__(self):
        self.rag = EnhancedRAGSystem()
        
    async def initialize(self):
        await self.rag.initialize(
            gpu_memory_utilization=0.87,
            max_num_batched_tokens=8192,
            max_num_seqs=64
        )
    
    async def single_request(self, query: str, method: str, request_id: int):
        """单个请求"""
        try:
            result = await self.rag.query(query, method=method, max_tokens=512)
            result['request_id'] = request_id
            result['success'] = True
            result['query'] = query
            return result
        except Exception as e:
            print(f"❌ 请求 {request_id} 失败: {e}")
            return {
                'request_id': request_id,
                'query': query,
                'success': False,
                'error': str(e)
            }
    
    async def run_benchmark(self, method: str):
        """运行压测"""
        print(f"\n{'='*80}")
        print(f"📊 测试方法: {method}")
        print(f"总请求数: {TOTAL_REQUESTS} | 并发数: {CONCURRENCY}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # 创建任务
        tasks = []
        for i in range(TOTAL_REQUESTS):
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            task = self.single_request(query, method, i)
            tasks.append(task)
        
        # 限制并发
        semaphore = asyncio.Semaphore(CONCURRENCY)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[bounded_task(task) for task in tasks])
        
        duration = time.time() - start_time
        
        print(f"✅ 测试完成，耗时: {duration:.2f}秒\n")
        
        return self.analyze_results(results, duration, method)
    
    def analyze_results(self, results, duration, method):
        """分析结果"""
        success_results = [r for r in results if r.get('success', False)]
        
        if not success_results:
            print("❌ 没有成功的请求")
            return None
        
        # 提取指标
        total_times = [r['total_time'] for r in success_results]
        retrieval_times = [r['retrieval_time'] for r in success_results]
        rerank_times = [r.get('rerank_time', 0) for r in success_results]
        generation_times = [r['generation_time'] for r in success_results]
        ttfts = [r['ttft'] for r in success_results]
        
        def percentile(data, p):
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        
        stats = {
            'method': method,
            'test_duration': duration,
            'total_requests': len(results),
            'successful_requests': len(success_results),
            'success_rate': len(success_results) / len(results) * 100,
            'qps': len(success_results) / duration,
            'total_time': {
                'mean': statistics.mean(total_times),
                'median': statistics.median(total_times),
                'min': min(total_times),
                'max': max(total_times),
                'p95': percentile(total_times, 95),
                'p99': percentile(total_times, 99)
            },
            'retrieval_time': {
                'mean': statistics.mean(retrieval_times),
                'median': statistics.median(retrieval_times),
                'p95': percentile(retrieval_times, 95)
            },
            'rerank_time': {
                'mean': statistics.mean([t for t in rerank_times if t > 0]) if any(rerank_times) else 0,
                'median': statistics.median([t for t in rerank_times if t > 0]) if any(rerank_times) else 0
            },
            'generation_time': {
                'mean': statistics.mean(generation_times),
                'median': statistics.median(generation_times),
                'p99': percentile(generation_times, 99)
            },
            'ttft': {
                'mean': statistics.mean(ttfts),
                'median': statistics.median(ttfts),
                'min': min(ttfts),
                'max': max(ttfts),
                'p95': percentile(ttfts, 95),
                'p99': percentile(ttfts, 99)
            }
        }
        
        self.print_stats(stats)
        return stats, success_results  # 返回原始结果用于质量评估
    
    def print_stats(self, stats):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"📊 性能统计 - {stats['method']}")
        print(f"{'='*80}")
        print(f"成功率: {stats['success_rate']:.1f}%")
        print(f"QPS: {stats['qps']:.2f}")
        print(f"\n【总响应时间】")
        print(f"  平均: {stats['total_time']['mean']:.3f}秒")
        print(f"  中位数: {stats['total_time']['median']:.3f}秒")
        print(f"  P95: {stats['total_time']['p95']:.3f}秒")
        print(f"  P99: {stats['total_time']['p99']:.3f}秒")
        print(f"\n【检索时间】")
        print(f"  平均: {stats['retrieval_time']['mean']:.3f}秒")
        print(f"  P95: {stats['retrieval_time']['p95']:.3f}秒")
        if stats['rerank_time']['mean'] > 0:
            print(f"\n【重排时间】")
            print(f"  平均: {stats['rerank_time']['mean']:.3f}秒")
            print(f"  中位数: {stats['rerank_time']['median']:.3f}秒")
        print(f"\n【生成时间】")
        print(f"  平均: {stats['generation_time']['mean']:.3f}秒")
        print(f"  P99: {stats['generation_time']['p99']:.3f}秒")
        print(f"\n【TTFT（首字延迟）】")
        print(f"  平均: {stats['ttft']['mean']:.3f}秒")
        print(f"  中位数: {stats['ttft']['median']:.3f}秒")
        print(f"  P95: {stats['ttft']['p95']:.3f}秒")
        print(f"  P99: {stats['ttft']['p99']:.3f}秒")
        print("="*80 + "\n")

async def main():
    """主函数"""
    print("\n" + "="*80)
    print("🎯 戒毒知识库RAG系统完整压测")
    print("="*80)
    
    benchmark = RAGBenchmark()
    await benchmark.initialize()
    
    # 测试三种方法
    methods = ["baseline", "rerank", "hybrid_rerank"]
    all_stats = []
    all_results_dict = {}
    
    for i, method in enumerate(methods, 1):
        print(f"\n[{i}/{len(methods)}] 开始测试: {method}")
        stats, results = await benchmark.run_benchmark(method)
        
        if stats:
            all_stats.append(stats)
            all_results_dict[method] = results
        
        # 等待GPU冷却
        if i < len(methods):
            print("⏳ 等待5秒让GPU冷却...\n")
            await asyncio.sleep(5)
    
    # 生成对比报告
    generate_comparison_report(all_stats, all_results_dict)

def generate_comparison_report(all_stats, all_results_dict):
    """生成对比报告"""
    print("\n" + "="*80)
    print("📊 方法对比总结")
    print("="*80 + "\n")
    
    if not all_stats:
        print("❌ 没有可用结果")
        return
    
    # 打印对比表格
    print(f"{'方法':<25} {'QPS':<10} {'平均响应':<12} {'P99响应':<12} {'P99 TTFT':<12}")
    print("-" * 80)
    
    baseline = all_stats[0]
    
    for stat in all_stats:
        print(f"{stat['method']:<25} "
              f"{stat['qps']:<10.2f} "
              f"{stat['total_time']['mean']:<12.3f} "
              f"{stat['total_time']['p99']:<12.3f} "
              f"{stat['ttft']['p99']:<12.3f}")
    
    # 相对提升分析
    print("\n【相对Baseline提升】\n")
    for stat in all_stats[1:]:
        qps_change = (stat['qps'] / baseline['qps'] - 1) * 100
        resp_change = (1 - stat['total_time']['mean'] / baseline['total_time']['mean']) * 100
        ttft_change = (1 - stat['ttft']['mean'] / baseline['ttft']['mean']) * 100
        
        print(f"方法: {stat['method']}")
        print(f"  QPS变化: {qps_change:+.1f}%")
        print(f"  响应时间变化: {resp_change:+.1f}%")
        print(f"  TTFT变化: {ttft_change:+.1f}%")
        print(f"  检索时间: {stat['retrieval_time']['mean']:.3f}秒")
        if stat['rerank_time']['mean'] > 0:
            print(f"  重排时间: {stat['rerank_time']['mean']:.3f}秒")
        print()
    
    # 保存JSON报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"jieduo_benchmark_report_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'test_date': timestamp,
                'knowledge_base': 'jieduo_policies',
                'total_docs': 9,
                'total_chars': 56364,
                'total_chunks': 142,
                'num_methods': len(all_stats),
                'requests_per_method': TOTAL_REQUESTS,
                'concurrency': CONCURRENCY
            },
            'performance_stats': all_stats
        }, f, indent=2, ensure_ascii=False)
    
    print(f"📄 性能报告已保存: {filename}")
    
    # 同时保存原始结果用于质量评估
    results_filename = f"jieduo_benchmark_results_{timestamp}.json"
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'results': {method: [
                {
                    'query': r['query'],
                    'answer': r['answer'],
                    'retrieval_method': r['retrieval_method']
                } for r in results
            ] for method, results in all_results_dict.items()}
        }, f, indent=2, ensure_ascii=False)
    
    print(f"📄 答案结果已保存: {results_filename}")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())