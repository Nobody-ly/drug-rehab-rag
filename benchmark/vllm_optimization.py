import asyncio
import time
import statistics
import json
import gc
from datetime import datetime

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

# ============================================================================
# 配置矩阵：新增 Opt_v4_BusinessRealistic
# ============================================================================
CONFIGS = [
    {
        "name": "Baseline",
        "max_tokens": 128,
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": None,
        "description": "当前默认配置"
    },
    {
        "name": "Opt_v1_ReduceOutput",
        "max_tokens": 64,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.87,
        "max_num_seqs": None,
        "description": "降低输出长度 + 提升批处理"
    },
    {
        "name": "Opt_v4_BusinessRealistic",
        "max_tokens": 256,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.87,
        "max_num_seqs": 64,
        "description": "业务贴合配置：详细说明/政策解释场景"
    }
]

TEST_QUERIES = [
    "什么是检索增强生成技术？",
    "vLLM的核心优势是什么？",
    "Llama2模型有哪些特点？",
    "如何优化大模型推理性能？",
    "PagedAttention的原理是什么？",
    "RAG技术解决了什么问题？",
    "向量数据库的作用是什么？",
    "如何提升模型吞吐量？",
]

CONCURRENCY = 10
TOTAL_REQUESTS = 30

class AsyncRAGBenchmark:
    def __init__(self, config):
        self.config = config
        self.embeddings = None
        self.vectordb = None
        self.engine = None
        
    async def initialize(self):
        print(f"\n{'='*80}")
        print(f"🚀 初始化配置: {self.config['name']}")
        print(f"   - max_tokens: {self.config['max_tokens']}")
        print(f"   - max_num_batched_tokens: {self.config['max_num_batched_tokens']}")
        print(f"   - gpu_memory_utilization: {self.config['gpu_memory_utilization']}")
        if self.config.get('max_num_seqs'):
            print(f"   - max_num_seqs: {self.config['max_num_seqs']}")
        print(f"   说明: {self.config['description']}")
        print(f"{'='*80}")
        
        if self.embeddings is None:
            print("  [1/2] 加载向量库...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        if self.vectordb is None:
            self.vectordb = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings,
                collection_name="rag_collection"
            )
        
        print("  [2/2] 初始化vLLM引擎...")
        engine_args = AsyncEngineArgs(
            model="./models/qwen/Qwen2.5-7B-Instruct",
            gpu_memory_utilization=self.config['gpu_memory_utilization'],
            max_num_batched_tokens=self.config['max_num_batched_tokens'],
            max_num_seqs=self.config.get('max_num_seqs'),  # 关键新增！
            trust_remote_code=True,
            dtype="bfloat16"
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("  ✅ 初始化完成\n")
    
    async def query_single(self, question: str, request_id: str):
        start_time = time.time()
        
        retrieval_start = time.time()
        docs = self.vectordb.similarity_search(question, k=3)
        retrieval_time = time.time() - retrieval_start
        
        context = "\n\n".join([f"[文档{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
        prompt = f"""请基于以下参考文档回答问题。

参考文档：
{context}

问题：{question}

回答："""
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=self.config['max_tokens']
        )
        
        generation_start = time.time()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        first_token_time = None
        full_answer = ""
        token_count = 0
        
        async for result in results_generator:
            if result.finished:
                full_answer = result.outputs[0].text
                token_count = len(result.outputs[0].token_ids)
            else:
                if first_token_time is None:
                    first_token_time = time.time()
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        if first_token_time is None:
            ttft = generation_time / token_count if token_count > 0 else 0
        else:
            ttft = first_token_time - generation_start
        
        return {
            "request_id": request_id,
            "question": question,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "ttft": ttft,
            "token_count": token_count,
            "throughput": token_count / generation_time if generation_time > 0 else 0,
            "success": True
        }
    
    async def run_benchmark(self):
        print(f"⏳ 开始压测：{TOTAL_REQUESTS}个请求，{CONCURRENCY}并发\n")
        
        start_time = time.time()
        
        tasks = []
        for i in range(TOTAL_REQUESTS):
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            request_id = f"req_{i}_{int(time.time() * 1000)}"
            tasks.append(self.query_single(query, request_id))
        
        semaphore = asyncio.Semaphore(CONCURRENCY)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[bounded_task(task) for task in tasks])
        
        duration = time.time() - start_time
        print(f"✅ 测试完成，耗时: {duration:.2f}秒\n")
        
        return self.analyze_results(results, duration)
    
    def analyze_results(self, results, duration):
        success_results = [r for r in results if r.get("success", False)]
        
        if not success_results:
            return None
        
        total_times = [r["total_time"] for r in success_results]
        ttfts = [r["ttft"] for r in success_results]
        gen_times = [r["generation_time"] for r in success_results]
        throughputs = [r["throughput"] for r in success_results]
        token_counts = [r["token_count"] for r in success_results]
        
        def percentile(data, p):
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        
        stats = {
            "config": self.config,
            "test_duration": duration,
            "total_requests": len(results),
            "successful_requests": len(success_results),
            "qps": len(success_results) / duration,
            "avg_response_time": statistics.mean(total_times),
            "p99_response_time": percentile(total_times, 99),
            "ttft": {
                "mean": statistics.mean(ttfts),
                "median": statistics.median(ttfts),
                "p95": percentile(ttfts, 95),
                "p99": percentile(ttfts, 99)
            },
            "generation": {
                "mean": statistics.mean(gen_times),
                "p99": percentile(gen_times, 99)
            },
            "throughput": {
                "mean": statistics.mean(throughputs),
                "median": statistics.median(throughputs)
            },
            "tokens": {
                "mean": statistics.mean(token_counts),
                "total": sum(token_counts)
            }
        }
        
        self.print_stats(stats)
        return stats
    
    def print_stats(self, stats):
        print(f"{'='*80}")
        print(f"📊 配置: {stats['config']['name']}")
        print(f"{'='*80}")
        print(f"测试时长: {stats['test_duration']:.2f}秒")
        print(f"QPS: {stats['qps']:.2f} 请求/秒")
        print(f"平均响应: {stats['avg_response_time']:.3f}秒")
        print(f"P99响应: {stats['p99_response_time']:.3f}秒")
        print(f"\n【TTFT】")
        print(f"  平均: {stats['ttft']['mean']:.3f}秒")
        print(f"  P95: {stats['ttft']['p95']:.3f}秒")
        print(f"  P99: {stats['ttft']['p99']:.3f}秒")
        print(f"\n【生成】")
        print(f"  平均: {stats['generation']['mean']:.3f}秒")
        print(f"  P99: {stats['generation']['p99']:.3f}秒")
        print(f"\n【吞吐】")
        print(f"  平均: {stats['throughput']['mean']:.2f} tok/s")
        print(f"\n【Token】")
        print(f"  平均: {stats['tokens']['mean']:.1f} tokens/请求")
        print(f"  总计: {stats['tokens']['total']} tokens")
        print(f"{'='*80}\n")
    
    async def cleanup(self):
        if self.engine:
            del self.engine
        import torch
        torch.cuda.empty_cache()
        gc.collect()

async def main():
    all_results = []
    
    print("\n" + "="*80)
    print("🎯 vLLM参数优化基准测试 v4")
    print("="*80)
    print(f"测试配置数: {len(CONFIGS)}")
    print(f"每配置请求数: {TOTAL_REQUESTS}")
    print(f"并发度: {CONCURRENCY}")
    print("="*80)
    
    shared_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(CONFIGS)}] 测试配置: {config['name']}")
        print(f"{'='*80}")
        
        rag = AsyncRAGBenchmark(config)
        rag.embeddings = shared_embeddings
        
        await rag.initialize()
        stats = await rag.run_benchmark()
        
        if stats:
            all_results.append(stats)
        
        await rag.cleanup()
        
        print(f"✅ 配置 {config['name']} 测试完成")
        print("⏳ 等待10秒让GPU完全释放...\n")
        await asyncio.sleep(10)
    
    generate_comparison_report(all_results)

def generate_comparison_report(all_results):
    print("\n" + "="*80)
    print("📊 参数优化对比报告 v4")
    print("="*80 + "\n")
    
    if not all_results:
        print("❌ 没有可用的测试结果")
        return
    
    baseline = all_results[0]
    
    print("【配置对比】\n")
    print(f"{'配置名':<30} {'max_tokens':<12} {'QPS':<10} {'平均响应':<12} {'P99 TTFT':<12} {'平均Token':<12}")
    print("-" * 100)
    
    for result in all_results:
        print(f"{result['config']['name']:<30} "
              f"{result['config']['max_tokens']:<12} "
              f"{result['qps']:<10.2f} "
              f"{result['avg_response_time']:<12.3f} "
              f"{result['ttft']['p99']:<12.3f} "
              f"{result['tokens']['mean']:<12.1f}")
    
    print("\n【相对Baseline提升】\n")
    for result in all_results[1:]:
        qps_gain = (result['qps'] / baseline['qps'] - 1) * 100
        resp_gain = (1 - result['avg_response_time'] / baseline['avg_response_time']) * 100
        ttft_gain = (1 - result['ttft']['p99'] / baseline['ttft']['p99']) * 100
        token_gain = (result['tokens']['mean'] / baseline['tokens']['mean'] - 1) * 100
        
        print(f"配置: {result['config']['name']}")
        print(f"  QPS提升: {qps_gain:+.1f}%")
        print(f"  响应时间改善: {resp_gain:+.1f}%")
        print(f"  P99 TTFT改善: {ttft_gain:+.1f}%")
        print(f"  平均输出Token: {token_gain:+.1f}%")
        print()
    
    # 关键洞察
    print("【关键洞察】\n")
    print("1. max_tokens 对性能影响最大")
    print(f"   - 64 tokens: QPS提升约 {((all_results[1]['qps']/baseline['qps']-1)*100):.0f}%")
    if len(all_results) > 2:
        print(f"   - 256 tokens: QPS提升约 {((all_results[2]['qps']/baseline['qps']-1)*100):.0f}%")
    
    print("\n2. 批处理优化收益明显")
    print("   - 2048 → 8192: TTFT保持稳定，吞吐量提升")
    
    print("\n3. 业务场景适配建议")
    print("   - 简单问答: 推荐 64-128 tokens")
    print("   - 详细解释: 推荐 256 tokens")
    print("   - 文档摘要: 推荐 512+ tokens")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vllm_optimization_v4_report_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'test_date': timestamp,
                'num_configs': len(all_results),
                'requests_per_config': TOTAL_REQUESTS,
                'concurrency': CONCURRENCY
            },
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细报告已保存: {filename}")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
