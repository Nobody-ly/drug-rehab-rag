import asyncio
import time
import statistics
import json
from datetime import datetime

# 导入异步RAG系统
import sys
sys.path.append('.')

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

# 压测配置（与load_test.py保持一致）
CONCURRENCY = 10  # 10并发
TOTAL_REQUESTS = 100  # 总请求数

# 测试query列表（与load_test.py保持一致）
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

class AsyncRAGSystemBenchmark:
    """异步RAG系统（用于压测）"""
    def __init__(self):
        print("🚀 初始化异步RAG系统...")
        
        # 1. 加载向量库
        print("  [1/3] 加载向量库...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectordb = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings,
            collection_name="rag_collection"
        )
        
        # 2. 初始化异步LLM引擎
        print("  [2/3] 初始化异步vLLM引擎...")
        engine_args = AsyncEngineArgs(
            model="./models/qwen/Qwen2.5-7B-Instruct",
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            dtype="bfloat16"
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        print("  [3/3] 系统初始化完成 ✅\n")
    
    async def query_stream(self, question: str, request_id: str, top_k: int = 3):
        """异步流式查询（支持真实TTFT测量）"""
        start_time = time.time()
        
        # Step 1: 向量检索
        retrieval_start = time.time()
        docs = self.vectordb.similarity_search(question, k=top_k)
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: 构建Prompt
        context = "\n\n".join([
            f"[文档{i+1}] {doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
        
        prompt = f"""请基于以下参考文档回答问题。

参考文档：
{context}

问题：{question}

回答："""
        
        # Step 3: 流式生成
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128
        )
        
        generation_start = time.time()
        
        # 添加请求到引擎
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id
        )
        
        # 流式接收token
        first_token_time = None
        full_answer = ""
        token_count = 0
        
        async for result in results_generator:
            if result.finished:
                # 最终输出
                full_answer = result.outputs[0].text
                token_count = len(result.outputs[0].token_ids)
            else:
                # 中间token（第一次进入此分支时记录TTFT）
                if first_token_time is None:
                    first_token_time = time.time()
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # 如果没有捕获到中间token，用估算值
        if first_token_time is None:
            ttft = generation_time / token_count if token_count > 0 else 0
        else:
            ttft = first_token_time - generation_start
        
        return {
            "request_id": request_id,
            "question": question,
            "answer": full_answer,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "ttft": ttft,
            "token_count": token_count,
            "throughput": token_count / generation_time if generation_time > 0 else 0,
            "success": True
        }

class AsyncLoadTester:
    def __init__(self):
        self.results = []
        self.errors = 0
        self.start_time = None
    
    async def run_single_request(self, rag_system, query_idx):
        """执行单个请求"""
        query = TEST_QUERIES[query_idx % len(TEST_QUERIES)]
        request_id = f"req_{query_idx}_{int(time.time() * 1000)}"
        
        try:
            result = await rag_system.query_stream(query, request_id)
            return result
        except Exception as e:
            self.errors += 1
            return {
                "success": False,
                "request_id": request_id,
                "question": query,
                "error": str(e)
            }
    
    async def run_load_test(self, num_requests, concurrency):
        """执行异步并发压测"""
        print("="*80)
        print(f"📊 异步流式RAG并发压测")
        print(f"   - 总请求数: {num_requests}")
        print(f"   - 并发数: {concurrency}")
        print(f"   - 查询类型: {len(TEST_QUERIES)}种")
        print("="*80 + "\n")
        
        # 初始化RAG系统（只初始化一次）
        rag_system = AsyncRAGSystemBenchmark()
        
        self.start_time = time.time()
        
        # 创建所有任务
        tasks = []
        for i in range(num_requests):
            task = self.run_single_request(rag_system, i)
            tasks.append(task)
        
        # 使用Semaphore控制并发度
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        # 执行所有任务
        print("⏳ 开始执行压测...\n")
        results = await asyncio.gather(*[bounded_task(task) for task in tasks])
        
        # 收集结果
        for result in results:
            if result.get("success", False):
                self.results.append(result)
        
        total_duration = time.time() - self.start_time
        
        print(f"\n✅ 压测完成，耗时: {total_duration:.2f}秒\n")
        
        # 生成报告
        self.generate_report(total_duration)
    
    def generate_report(self, total_duration):
        """生成性能报告"""
        if not self.results:
            print("❌ 没有成功的请求！")
            return
        
        # 提取时间数据
        total_times = [r["total_time"] for r in self.results]
        retrieval_times = [r["retrieval_time"] for r in self.results]
        generation_times = [r["generation_time"] for r in self.results]
        ttfts = [r["ttft"] for r in self.results]
        throughputs = [r["throughput"] for r in self.results]
        
        # 计算统计指标
        def calc_percentile(data, percentile):
            sorted_data = sorted(data)
            index = int(len(sorted_data) * percentile / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]
        
        report = {
            "test_config": {
                "total_requests": len(self.results) + self.errors,
                "successful_requests": len(self.results),
                "failed_requests": self.errors,
                "concurrency": CONCURRENCY,
                "duration": f"{total_duration:.2f}s"
            },
            "throughput": {
                "qps": len(self.results) / total_duration,
                "avg_response_time": statistics.mean(total_times),
                "avg_token_throughput": statistics.mean(throughputs)
            },
            "latency": {
                "total_time": {
                    "min": min(total_times),
                    "max": max(total_times),
                    "mean": statistics.mean(total_times),
                    "median": statistics.median(total_times),
                    "p95": calc_percentile(total_times, 95),
                    "p99": calc_percentile(total_times, 99)
                },
                "ttft": {
                    "min": min(ttfts),
                    "max": max(ttfts),
                    "mean": statistics.mean(ttfts),
                    "median": statistics.median(ttfts),
                    "p95": calc_percentile(ttfts, 95),
                    "p99": calc_percentile(ttfts, 99)
                },
                "retrieval_time": {
                    "mean": statistics.mean(retrieval_times),
                    "p99": calc_percentile(retrieval_times, 99)
                },
                "generation_time": {
                    "mean": statistics.mean(generation_times),
                    "p99": calc_percentile(generation_times, 99)
                }
            }
        }
        
        # 打印报告
        print("="*80)
        print("📊 异步流式RAG压测结果")
        print("="*80 + "\n")
        
        print("【测试配置】")
        print(f"  总请求数: {report['test_config']['total_requests']}")
        print(f"  成功请求: {report['test_config']['successful_requests']}")
        print(f"  失败请求: {report['test_config']['failed_requests']}")
        print(f"  并发数: {report['test_config']['concurrency']}")
        print(f"  测试时长: {report['test_config']['duration']}\n")
        
        print("【吞吐量指标】")
        print(f"  QPS: {report['throughput']['qps']:.2f} 请求/秒")
        print(f"  平均响应时间: {report['throughput']['avg_response_time']:.3f}秒")
        print(f"  平均token吞吐: {report['throughput']['avg_token_throughput']:.2f} tok/s\n")
        
        print("【延迟指标 - TTFT（关键指标）】")
        print(f"  最小TTFT: {report['latency']['ttft']['min']:.3f}秒")
        print(f"  最大TTFT: {report['latency']['ttft']['max']:.3f}秒")
        print(f"  平均TTFT: {report['latency']['ttft']['mean']:.3f}秒")
        print(f"  中位TTFT: {report['latency']['ttft']['median']:.3f}秒")
        print(f"  P95 TTFT: {report['latency']['ttft']['p95']:.3f}秒")
        print(f"  P99 TTFT: {report['latency']['ttft']['p99']:.3f}秒\n")
        
        print("【延迟指标 - 端到端】")
        print(f"  最小值: {report['latency']['total_time']['min']:.3f}秒")
        print(f"  最大值: {report['latency']['total_time']['max']:.3f}秒")
        print(f"  平均值: {report['latency']['total_time']['mean']:.3f}秒")
        print(f"  中位数: {report['latency']['total_time']['median']:.3f}秒")
        print(f"  P95: {report['latency']['total_time']['p95']:.3f}秒")
        print(f"  P99: {report['latency']['total_time']['p99']:.3f}秒\n")
        
        print("【延迟指标 - 检索】")
        print(f"  平均检索时间: {report['latency']['retrieval_time']['mean']:.3f}秒")
        print(f"  P99检索时间: {report['latency']['retrieval_time']['p99']:.3f}秒\n")
        
        print("【延迟指标 - 生成】")
        print(f"  平均生成时间: {report['latency']['generation_time']['mean']:.3f}秒")
        print(f"  P99生成时间: {report['latency']['generation_time']['p99']:.3f}秒\n")
        
        # 保存JSON报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"async_load_test_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'report': report,
                'raw_results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细报告已保存: {filename}")
        print("="*80 + "\n")
        
        return report

async def main():
    """主函数"""
    tester = AsyncLoadTester()
    await tester.run_load_test(
        num_requests=TOTAL_REQUESTS,
        concurrency=CONCURRENCY
    )

if __name__ == "__main__":
    asyncio.run(main())