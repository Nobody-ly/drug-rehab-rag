import requests
import time
import statistics
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 压测配置
API_URL = "http://localhost:8000/query"
CONCURRENCY = 10  # 10并发
TOTAL_REQUESTS = 100  # 总请求数
DURATION = 300  # 5分钟压测

# 测试query列表
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

class LoadTester:
    def __init__(self):
        self.results = []
        self.errors = 0
        self.start_time = None
        
    def send_request(self, query_idx):
        """发送单个请求"""
        query = TEST_QUERIES[query_idx % len(TEST_QUERIES)]
        
        try:
            start = time.time()
            response = requests.post(
                API_URL,
                json={"question": query, "top_k": 3},
                timeout=30
            )
            end = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "query": query,
                    "total_time": end - start,
                    "retrieval_time": data["retrieval_time"],
                    "generation_time": data["generation_time"],
                    "ttft": None  # API层面无法直接测量TTFT
                }
            else:
                self.errors += 1
                return {"success": False, "error": response.status_code}
                
        except Exception as e:
            self.errors += 1
            return {"success": False, "error": str(e)}
    
    def run_load_test(self, num_requests, concurrency):
        """执行压测"""
        print("="*80)
        print(f"📊 开始压测")
        print(f"   - 总请求数: {num_requests}")
        print(f"   - 并发数: {concurrency}")
        print(f"   - 查询类型: {len(TEST_QUERIES)}种")
        print("="*80 + "\n")
        
        self.start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.send_request, i) 
                for i in range(num_requests)
            ]
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    self.results.append(result)
                
                completed += 1
                if completed % 10 == 0:
                    print(f"⏳ 已完成: {completed}/{num_requests} 请求")
        
        total_duration = time.time() - self.start_time
        
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
                "avg_response_time": statistics.mean(total_times)
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
        print("\n" + "="*80)
        print("📊 压测结果报告")
        print("="*80 + "\n")
        
        print("【测试配置】")
        print(f"  总请求数: {report['test_config']['total_requests']}")
        print(f"  成功请求: {report['test_config']['successful_requests']}")
        print(f"  失败请求: {report['test_config']['failed_requests']}")
        print(f"  并发数: {report['test_config']['concurrency']}")
        print(f"  测试时长: {report['test_config']['duration']}\n")
        
        print("【吞吐量指标】")
        print(f"  QPS: {report['throughput']['qps']:.2f} 请求/秒")
        print(f"  平均响应时间: {report['throughput']['avg_response_time']:.3f}秒\n")
        
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
        filename = f"load_test_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'report': report,
                'raw_results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细报告已保存: {filename}")
        print("="*80 + "\n")
        
        return report

if __name__ == "__main__":
    # 等待API服务器启动
    print("⏳ 等待API服务器就绪...")
    time.sleep(5)
    
    # 健康检查
    try:
        response = requests.get("http://localhost:8000/")
        print(f"✅ API服务器状态: {response.json()}\n")
    except Exception as e:
        print(f"❌ 无法连接到API服务器: {e}")
        print("请先启动API服务器: python api_server.py")
        exit(1)
    
    # 执行压测
    tester = LoadTester()
    tester.run_load_test(
        num_requests=100,
        concurrency=10
    )