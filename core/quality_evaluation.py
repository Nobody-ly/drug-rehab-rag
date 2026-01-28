import json
import asyncio
from datetime import datetime

# 质量评估问题集（带标准答案）
EVALUATION_SET = [
    {
        "question": "什么是强制隔离戒毒？",
        "key_points": [
            "公安机关作出决定",
            "戒毒场所",
            "强制性教育矫治",
            "适用条件"
        ],
        "answer_type": "定义"
    },
    {
        "question": "社区戒毒的适用条件是什么？",
        "key_points": [
            "首次吸毒",
            "成瘾程度较轻",
            "本人自愿",
            "县级公安机关决定"
        ],
        "answer_type": "条件"
    },
    {
        "question": "戒毒人员有哪些权利和保障？",
        "key_points": [
            "不受歧视",
            "个人信息保密",
            "入学就业",
            "享受社会保障"
        ],
        "answer_type": "权利"
    },
    {
        "question": "强制隔离戒毒的期限是多久？",
        "key_points": [
            "二年",
            "可以延长一年",
            "最长不超过三年"
        ],
        "answer_type": "事实"
    },
    {
        "question": "戒毒场所应当提供哪些服务？",
        "key_points": [
            "戒毒治疗",
            "心理治疗",
            "身体康复训练",
            "职业技能培训",
            "法制教育"
        ],
        "answer_type": "列举"
    }
]

def evaluate_answer_quality(question_data, answer):
    """
    评估答案质量
    
    评分标准（5分制）：
    - 5分：包含所有关键点，表述清晰准确
    - 4分：包含大部分关键点，表述基本准确
    - 3分：包含部分关键点，表述有缺失
    - 2分：仅包含少量关键点，表述不完整
    - 1分：基本不相关或错误
    """
    key_points = question_data['key_points']
    
    # 统计包含的关键点数量
    matched_points = 0
    for point in key_points:
        if point in answer:
            matched_points += 1
    
    # 计算覆盖率
    coverage_rate = matched_points / len(key_points)
    
    # 评分
    if coverage_rate >= 0.8:
        score = 5
    elif coverage_rate >= 0.6:
        score = 4
    elif coverage_rate >= 0.4:
        score = 3
    elif coverage_rate >= 0.2:
        score = 2
    else:
        score = 1
    
    return {
        'score': score,
        'matched_points': matched_points,
        'total_points': len(key_points),
        'coverage_rate': coverage_rate
    }

def evaluate_from_results_file(results_file):
    """从压测结果文件中评估质量"""
    print("\n" + "="*80)
    print("📊 RAG系统质量评估")
    print("="*80 + "\n")
    
    # 加载压测结果
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 对每种方法进行评估
    method_scores = {}
    
    for method, answers in results.items():
        print(f"\n{'='*80}")
        print(f"评估方法: {method}")
        print(f"{'='*80}\n")
        
        scores = []
        detailed_results = []
        
        for eval_q in EVALUATION_SET:
            question = eval_q['question']
            
            # 找到对应的答案
            answer_obj = next((a for a in answers if a['query'] == question), None)
            
            if not answer_obj:
                print(f"⚠️  未找到问题的答案: {question}")
                continue
            
            answer = answer_obj['answer']
            
            # 评估
            eval_result = evaluate_answer_quality(eval_q, answer)
            scores.append(eval_result['score'])
            
            detailed_results.append({
                'question': question,
                'score': eval_result['score'],
                'coverage': eval_result['coverage_rate'],
                'matched': eval_result['matched_points'],
                'total': eval_result['total_points']
            })
            
            print(f"问题: {question}")
            print(f"得分: {eval_result['score']}/5")
            print(f"覆盖率: {eval_result['coverage_rate']:.1%}")
            print(f"关键点: {eval_result['matched_points']}/{eval_result['total_points']}")
            print()
        
        # 计算平均分
        avg_score = sum(scores) / len(scores) if scores else 0
        
        method_scores[method] = {
            'average_score': avg_score,
            'total_questions': len(scores),
            'detailed_results': detailed_results
        }
        
        print(f"📊 {method} 平均得分: {avg_score:.2f}/5 ({avg_score/5*100:.1f}%)\n")
    
    # 对比分析
    print("\n" + "="*80)
    print("📊 质量对比总结")
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
        
        print(f"\n✨ Hybrid+Reranker 相对 Baseline 准确率提升: {improvement:+.1f}%")
    
    # 保存评估报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quality_evaluation_report_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'evaluation_set_size': len(EVALUATION_SET),
            'method_scores': method_scores
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 质量评估报告已保存: {filename}")
    print("="*80 + "\n")
    
    return method_scores

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python quality_evaluation.py <压测结果文件>")
        print("示例: python quality_evaluation.py jieduo_benchmark_results_20260128_103000.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    evaluate_from_results_file(results_file)