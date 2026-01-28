import asyncio
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # ✅ 修复：使用langchain_core
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import jieba

@dataclass
class RetrievalResult:
    """检索结果数据类"""
    docs: List[Document]
    method: str
    retrieval_time: float
    rerank_time: float = 0.0

class EnhancedRAGSystem:
    """增强版RAG系统：Hybrid Search + Reranker"""
    
    def __init__(self, 
                 vector_db_path: str = "./chroma_db_jieduo",
                 embedding_model: str = "BAAI/bge-m3",
                 reranker_model: str = "BAAI/bge-reranker-base",
                 llm_model: str = "./models/qwen/Qwen2.5-7B-Instruct"):
        
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.llm_model = llm_model
        
        # 初始化组件
        self.embeddings = None
        self.vectordb = None
        self.reranker = None
        self.bm25 = None
        self.documents = None
        self.engine = None
        
        print("📦 EnhancedRAGSystem 初始化完成")
    
    async def initialize(self, 
                        gpu_memory_utilization: float = 0.87,
                        max_num_batched_tokens: int = 8192,
                        max_num_seqs: int = 64):
        """异步初始化所有组件"""
        
        print("\n" + "="*80)
        print("🚀 开始初始化增强版RAG系统")
        print("="*80)
        
        # 1. 加载Embedding模型
        print("\n[1/5] 加载Embedding模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embedding模型加载完成")
        
        # 2. 加载向量数据库
        print("\n[2/5] 加载向量数据库...")
        self.vectordb = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embeddings,
            collection_name="jieduo_collection"
        )
        
        # 获取所有文档用于BM25
        all_data = self.vectordb.get()
        self.documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_data['documents'], all_data['metadatas'])
        ]
        print(f"✅ 向量库加载完成，共 {len(self.documents)} 篇文档")
        
        # 3. 初始化BM25
        print("\n[3/5] 初始化BM25检索器...")
        tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ BM25初始化完成")
        
        # 4. 加载Reranker模型
        print("\n[4/5] 加载Reranker模型...")
        self.reranker = CrossEncoder(self.reranker_model_name, max_length=512, device='cuda')
        print("✅ Reranker模型加载完成")
        
        # 5. 初始化vLLM引擎
        print("\n[5/5] 初始化vLLM引擎...")
        engine_args = AsyncEngineArgs(
            model=self.llm_model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
            dtype="bfloat16"
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("✅ vLLM引擎初始化完成")
        
        print("\n" + "="*80)
        print("🎉 所有组件初始化完成！")
        print("="*80 + "\n")
    
    def _vector_search(self, query: str, k: int = 10) -> List[Document]:
        """向量检索（语义相似）"""
        return self.vectordb.similarity_search(query, k=k)
    
    def _bm25_search(self, query: str, k: int = 10) -> List[Document]:
        """BM25检索（关键词匹配）"""
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top-k的索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        return [self.documents[i] for i in top_indices]
    
    def _hybrid_search(self, query: str, k: int = 20) -> List[Document]:
        """混合检索：向量 + BM25"""
        # 各检索k/2个
        vector_docs = self._vector_search(query, k=k//2)
        bm25_docs = self._bm25_search(query, k=k//2)
        
        # 合并去重（保持顺序）
        seen = set()
        merged = []
        for doc in vector_docs + bm25_docs:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                merged.append(doc)
        
        return merged[:k]
    
    def _rerank(self, query: str, docs: List[Document], top_k: int = 3) -> Tuple[List[Document], List[float]]:
        """Reranker重排序"""
        if not docs:
            return [], []
        
        # 构造query-doc对
        pairs = [[query, doc.page_content] for doc in docs]
        
        # 打分
        scores = self.reranker.predict(pairs)
        
        # 排序
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
        
        ranked_docs = [doc for doc, score in ranked]
        ranked_scores = [score for doc, score in ranked]
        
        return ranked_docs, ranked_scores
    
    async def retrieve_baseline(self, query: str, k: int = 3) -> RetrievalResult:
        """Baseline：纯向量检索"""
        start = time.time()
        docs = self._vector_search(query, k=k)
        retrieval_time = time.time() - start
        
        return RetrievalResult(
            docs=docs,
            method="Baseline (Vector Only)",
            retrieval_time=retrieval_time
        )
    
    async def retrieve_with_rerank(self, query: str, k: int = 3) -> RetrievalResult:
        """方案1：向量检索 + Reranker"""
        start = time.time()
        candidates = self._vector_search(query, k=20)
        retrieval_time = time.time() - start
        
        rerank_start = time.time()
        docs, scores = self._rerank(query, candidates, top_k=k)
        rerank_time = time.time() - rerank_start
        
        return RetrievalResult(
            docs=docs,
            method="Vector + Reranker",
            retrieval_time=retrieval_time,
            rerank_time=rerank_time
        )
    
    async def retrieve_hybrid_rerank(self, query: str, k: int = 3) -> RetrievalResult:
        """方案2：混合检索 + Reranker（最优）"""
        start = time.time()
        candidates = self._hybrid_search(query, k=20)
        retrieval_time = time.time() - start
        
        rerank_start = time.time()
        docs, scores = self._rerank(query, candidates, top_k=k)
        rerank_time = time.time() - rerank_start
        
        return RetrievalResult(
            docs=docs,
            method="Hybrid + Reranker",
            retrieval_time=retrieval_time,
            rerank_time=rerank_time
        )
    
    async def query(self, 
                   question: str, 
                   method: str = "hybrid_rerank",
                   max_tokens: int = 128) -> Dict:
        """
        完整问答流程
        
        Args:
            question: 用户问题
            method: 检索方法 ["baseline", "rerank", "hybrid_rerank"]
            max_tokens: 最大生成token数
        """
        start_time = time.time()
        
        # 1. 检索
        if method == "baseline":
            retrieval_result = await self.retrieve_baseline(question)
        elif method == "rerank":
            retrieval_result = await self.retrieve_with_rerank(question)
        elif method == "hybrid_rerank":
            retrieval_result = await self.retrieve_hybrid_rerank(question)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 2. 构造prompt
        context = "\n\n".join([
            f"[文档{i+1}] {doc.page_content}" 
            for i, doc in enumerate(retrieval_result.docs)
        ])
        
#         prompt = f"""请基于以下参考文档回答问题。

# 参考文档：
# {context}

# 问题：{question}

# 回答："""
        prompt = f"""<|im_start|>system
你是一个专业的戒毒政策咨询助手。请基于提供的参考文档，准确、简洁地回答用户的问题。

【回答要求】
1. 直接回答问题，不要解释思考过程
2. 仅根据参考文档回答，不要编造信息
3. 保持简洁，控制在150-250字
4. 如果需要列举，使用数字列表

【参考文档】
{context}

【问题】
{question}

【回答】<|im_end|>
<|im_start|>assistant
"""
        
        # 3. 生成回答
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
            stop_token_ids=[151643, 151645],  # Qwen2.5 的 EOS token
            stop=["</s>", "<|im_end|>", "\n\n参考来源：", "参考文献：", "\n\n问题："],
            # skip_special_tokens=False,  # 保留特殊token检测
        )
        
        generation_start = time.time()
        request_id = f"req_{int(time.time() * 1000)}"
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        first_token_time = None
        full_answer = ""
        
        async for result in results_generator:
            if result.finished:
                full_answer = result.outputs[0].text
            else:
                if first_token_time is None:
                    first_token_time = time.time()
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        if first_token_time is None:
            ttft = 0.01
        else:
            ttft = first_token_time - generation_start
        
        return {
            "question": question,
            "answer": full_answer,
            "retrieval_method": retrieval_result.method,
            "retrieval_time": retrieval_result.retrieval_time,
            "rerank_time": retrieval_result.rerank_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "ttft": ttft,
            "retrieved_docs": len(retrieval_result.docs)
        }

# 测试函数
async def test_enhanced_rag():
    """快速测试"""
    print("\n" + "="*80)
    print("🧪 增强版RAG系统快速测试")
    print("="*80)
    
    # 初始化
    rag = EnhancedRAGSystem()
    await rag.initialize()
    
    # 测试问题
    test_queries = [
        "什么是检索增强生成技术？",
        "vLLM的核心优势是什么？",
        "如何优化大模型推理性能？"
    ]
    
    # 测试三种方法
    methods = ["baseline", "rerank", "hybrid_rerank"]
    
    for query in test_queries[:1]:  # 只测试第一个问题
        print(f"\n{'='*80}")
        print(f"问题: {query}")
        print(f"{'='*80}")
        
        for method in methods:
            result = await rag.query(query, method=method, max_tokens=64)
            
            print(f"\n【{result['retrieval_method']}】")
            print(f"  检索耗时: {result['retrieval_time']:.3f}秒")
            if result['rerank_time'] > 0:
                print(f"  重排耗时: {result['rerank_time']:.3f}秒")
            print(f"  生成耗时: {result['generation_time']:.3f}秒")
            print(f"  总耗时: {result['total_time']:.3f}秒")
            print(f"  TTFT: {result['ttft']:.3f}秒")
            print(f"  回答: {result['answer'][:100]}...")
    
    print("\n✅ 测试完成！\n")

if __name__ == "__main__":
    asyncio.run(test_enhanced_rag())