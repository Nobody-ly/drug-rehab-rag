from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 新版导入
import os

def load_and_split_documents(pdf_dir="./data/pdfs"):
    """加载PDF并分块"""
    documents = []
    
    print("📚 正在加载PDF文档...")
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_dir, filename)
            
            # 读取PDF
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            documents.append({
                'content': text,
                'source': filename
            })
            print(f"  ✓ {filename}: {len(text)} 字符")
    
    # 文档分块
    print("\n✂️  正在分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        splits = text_splitter.split_text(doc['content'])
        for i, chunk in enumerate(splits):
            chunks.append({
                'content': chunk,
                'source': doc['source'],
                'chunk_id': i
            })
    
    print(f"✅ 分块完成：总计 {len(chunks)} 个chunks")
    return chunks

# 测试
if __name__ == "__main__":
    chunks = load_and_split_documents()
    if chunks:
        print(f"\n预览第1个chunk:\n{chunks[0]['content'][:200]}...")
    else:
        print("⚠️  没有找到PDF文件，请先下载测试PDF")