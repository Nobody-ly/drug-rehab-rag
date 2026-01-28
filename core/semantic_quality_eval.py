import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 使用你已有的 BGE-M3 模型
model = SentenceTransformer('BAAI/bge-m3')

# 升级后的评估集（用语义相似度而非精确匹配）
EVALUATION_SET = [
    {
        "question": "什么是强制隔离戒毒？",
        "reference_answer": """
强制隔离戒毒是对吸毒成瘾人员在特定场所进行强制性戒毒治疗的措施。
由公安机关作出决定，适用于拒绝社区戒毒、在社区戒毒期间吸毒、
或吸毒成瘾严重难以通过社区戒毒戒除的人员。
不满16周岁未成年人、孕妇、哺乳期妇女不适用。
        """.strip(),
        "key_aspects": [
            "公安机关决定的强制措施",
            "在特定场所进行戒毒治疗",
            "适用条件：拒绝社区戒毒或吸毒严重",
            "特殊人群不适用：未成年人、孕妇等"
        ]
    },
    {
        "question": "社区戒毒的适用条件是什么？",
        "reference_answer": """
社区戒毒适用于被公安机关在作出行政处罚的同时责令接受社区戒毒的人员，
以及强制隔离戒毒人员因患严重疾病、健康状况不再适宜强制隔离戒毒的，
公安机关可以变更戒毒措施责令其接受社区戒毒。
        """.strip(),
        "key_aspects": [
            "行政处罚同时责令接受",
            "健康原因变更强制隔离措施",
            "由公安机关决定"
        ]
    },
    {
        "question": "戒毒人员有哪些权利和保障？",
        "reference_answer": """
戒毒人员在入学、就业、享受社会保障等方面不受歧视。
个人信息应当依法予以保密。
县级以上政府教育、民政、人力资源社会保障部门应当在入学、就业、
社会保障等方面对戒毒人员给予必要的指导和帮助。
戒断3年未复吸的人员，不再实行动态管控。
        """.strip(),
        "key_aspects": [
            "入学就业不受歧视",
            "个人信息保密",
            "政府部门给予指导帮助",
            "戒断3年后不再管控"
        ]
    },
    {
        "question": "强制隔离戒毒的期限是多久？",
        "reference_answer": """
强制隔离戒毒的期限为二年。
对于戒毒情况良好的，可以提前解除。
对于需要延长的，最长可以延长一年，即总期限最长为三年。
        """.strip(),
        "key_aspects": [
            "基本期限二年",
            "可提前解除",
            "最长延长一年至三年"
        ]
    },
    {
        "question": "戒毒场所应当提供哪些服务？",
        "reference_answer": """
戒毒场所应当提供戒毒康复指导、心理干预等专业服务。
提供戒毒医疗服务、心理康复、行为矫正、社会功能恢复等措施。
开展艾滋病等传染病的检测和预防教育。
提供必要的工作条件和保障。
        """.strip(),
        "key_aspects": [
            "戒毒康复指导",
            "心理干预和心理康复",
            "行为矫正和社会功能恢复",
            "医疗服务和传染病检测",
            "职业技能培训"
        ]
    }
]

def semantic_evaluate(answer, reference, key_aspects):
    """基于语义相似度的评估"""
    
    # 1. 整体相似度（40%权重）
    answer_emb = model.encode([answer])[0]
    ref_emb = model.encode([reference])[0]
    overall_sim = cosine_similarity([answer_emb], [ref_emb])[0][0]
    
    # 2. 关键方面覆盖度（60%权重）
    aspect_embs = model.encode(key_aspects)
    aspect_sims = cosine_similarity([answer_emb], aspect_embs)[0]
    
    # 每个关键方面的得分（相似度>0.5算覆盖）
    aspect_scores = [1 if sim > 0.5 else sim for sim in aspect_sims]
    aspect_coverage = np.mean(aspect_scores)
    
    # 综合得分
    final_score = 0.4 * overall_sim + 0.6 * aspect_coverage
    
    return {
        'score': final_score * 5,  # 转换为5分制
        'overall_similarity': overall_sim,
        'aspect_coverage': aspect_coverage,
        'aspect_details': list(zip(key_aspects, aspect_sims))
    }

def evaluate_from_file(results_file):
    """从压测结果文件评估"""
    print("\n" + "="*80)
    print("📊 语义质量评估（基于 BGE-M3 Embedding）")
    print("="*80 + "\n")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    method_scores = {}
    
    for method, answers in results.items():
        print(f"\n{'='*80}")
        print(f"评估方法: {method}")
        print(f"{'='*80}\n")
        
        scores = []
        detailed_results = []
        
        for eval_q in EVALUATION_SET:
            question = eval_q['question']
            answer_obj = next((a for a in answers if a['query'] == question), None)
            
            if not answer_obj:
                print(f"⚠️  未找到问题的答案: {question}")
                continue
            
            answer = answer_obj['answer']
            eval_result = semantic_evaluate(
                answer,
                eval_q['reference_answer'],
                eval_q['key_aspects']
            )
            
            scores.append(eval_result['score'])
            detailed_results.append({
                'question': question,
                'score': eval_result['score'],
                'overall_sim': eval_result['overall_similarity'],
                'aspect_cov': eval_result['aspect_coverage']
            })
            
            print(f"问题: {question}")
            print(f"得分: {eval_result['score']:.2f}/5")
            print(f"整体相似度: {eval_result['overall_similarity']:.2%}")
            print(f"关键点覆盖: {eval_result['aspect_coverage']:.2%}")
            print()
        
        avg_score = np.mean(scores) if scores else 0
        method_scores[method] = {
            'average_score': avg_score,
            'total_questions': len(scores),
            'detailed_results': detailed_results
        }
        
        print(f"📊 {method} 平均得分: {avg_score:.2f}/5 ({avg_score/5*100:.1f}%)\n")
    
    # 对比分析
    print("\n" + "="*80)
    print("📊 质量对比总结（语义评估）")
    print("="*80 + "\n")
    
    print(f"{'方法':<25} {'平均得分':<15} {'准确率':<15}")
    print("-" * 60)
    
    for method, data in method_scores.items():
        accuracy = data['average_score'] / 5 * 100
        print(f"{method:<25} {data['average_score']:.2f}/5{'':<9} {accuracy:.1f}%")
    
    # 计算提升
    if 'baseline' in method_scores and 'hybrid_rerank' in method_scores:
        baseline_score = method_scores['baseline']['average_score']
        hybrid_score = method_scores['hybrid_rerank']['average_score']
        improvement = (hybrid_score - baseline_score) / baseline_score * 100
        
        print(f"\n✨ Hybrid+Reranker 相对 Baseline 质量提升: {improvement:+.1f}%")
    
    # 保存报告
    timestamp = data['timestamp']
    filename = f"semantic_quality_report_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluation_method': 'semantic_similarity',
            'model': 'BAAI/bge-m3',
            'timestamp': timestamp,
            'method_scores': method_scores
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 语义评估报告已保存: {filename}")
    print("="*80 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python semantic_quality_eval.py <压测结果文件>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    evaluate_from_file(results_file)