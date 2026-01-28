import os
import shutil
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def load_documents_from_directory(directory):
    """加载txt和pdf文件"""
    print(f"\n[加载文档] 目录: {directory}")
    
    documents = []
    
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        
        try:
            if filename.endswith('.txt'):
                print(f"  📄 加载TXT: {filename}")
                loader = TextLoader(filepath, encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"     ✅ 成功")
                
            elif filename.endswith('.pdf'):
                print(f"  📕 加载PDF: {filename}")
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
                print(f"     ✅ 提取了 {len(docs)} 页")
                
        except Exception as e:
            print(f"     ❌ 加载失败: {e}")
            continue
    
    print(f"\n✅ 总共加载 {len(documents)} 个文档片段")
    return documents

def clean_text(text):
    """简单清理文本"""
    import re
    
    # 移除多余空行
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    
    # 移除明显的页码
    text = re.sub(r'第\s*\d+\s*页', '', text)
    text = re.sub(r'Page\s+\d+', '', text)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    return text

def build_vectordb():
    print("\n" + "="*80)
    print("🔨 构建戒毒政策知识库")
    print("="*80)
    
    # 1. 加载文档
    documents = load_documents_from_directory('data/jiedu')
    
    if not documents:
        print("❌ 错误：没有找到任何文档！")
        return
    
    # 2. 清理文本
    print("\n[清理文本] 移除页码和多余空行...")
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    
    # 3. 统计
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"✅ 文档总字数: {total_chars:,} 字")
    
    # 4. 分割文本
    print("\n[分割文本] 分块处理...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"✅ 分割成 {len(splits)} 个文本块")
    
    # 5. 加载Embedding（修复：使用本地模型，禁用在线检查）
    print("\n[加载模型] BAAI/bge-m3（本地模式）...")
    
    # 设置离线模式
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={
            'device': 'cuda',
            'trust_remote_code': True  # 信任本地代码
        },
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embedding模型加载完成")
    
    # 6. 构建向量库
    print("\n[构建向量库] 计算向量并存储...")
    
    if os.path.exists("chroma_db_jieduo"):
        shutil.rmtree("chroma_db_jieduo")
        print("🗑️  已删除旧向量库")
    
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db_jieduo",
        collection_name="jieduo_collection"
    )
    
    print("✅ 向量库构建完成！")
    
    # 7. 测试检索
    print("\n" + "="*80)
    print("🧪 测试检索功能")
    print("="*80)
    
    test_queries = [
        "什么是强制隔离戒毒？",
        "社区戒毒的条件是什么？",
        "戒毒人员有哪些权利？"
    ]
    
    for query in test_queries:
        print(f"\n问题: {query}")
        results = vectordb.similarity_search(query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"\n  [结果{i}]")
            print(f"  内容: {doc.page_content[:120]}...")
            source = doc.metadata.get('source', 'unknown')
            if '/' in source:
                source = source.split('/')[-1]
            print(f"  来源: {source}")
    
    print("\n" + "="*80)
    print("✅ 知识库已就绪！")
    print(f"📊 统计信息:")
    print(f"   - 向量库路径: ./chroma_db_jieduo")
    print(f"   - 原始文档数: {len(documents)}")
    print(f"   - 文本块数: {len(splits)}")
    print(f"   - 总字数: {total_chars:,}")
    print("="*80 + "\n")

if __name__ == "__main__":
    build_vectordb()