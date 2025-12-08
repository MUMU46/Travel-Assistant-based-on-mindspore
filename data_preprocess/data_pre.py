import json
import pandas as pd

def format_attrs(attrs):
    """将结构化属性转换为自然语言背景知识"""
    if not attrs: return ""
    return "\n".join([f"{a['name']}的{a['attrname']}是{a['attrvalue']}" for a in attrs])

def process_data(input_file, output_file):
    print(f"正在处理数据: {input_file} ...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    
    for session in data:
        messages = session['messages']
        
        # 遍历对话，寻找 Assistant (奇数索引) 的回复作为训练目标
        for i, msg in enumerate(messages):
            # 必须是 Assistant 的回合 (索引 1, 3, 5...)
            if i % 2 == 0: continue 
            
            # 获取上一句 User 的问题
            user_query = messages[i-1]['message']
            assistant_response = msg['message']
            
            # 构造 System 和 User 输入
            # 模式 A: 有知识库属性 (RAG模式)
            if 'attrs' in msg and msg['attrs']:
                context = format_attrs(msg['attrs'])
                instruction = (
                    "你是一个专业的北京旅游咨询助手。请基于以下【背景知识】回答用户问题。\n"
                    "回答风格要自然、亲切，但严禁编造事实。"
                )
                user_input = f"【背景知识】:\n{context}\n\n用户问题: {user_query}"
            
            # 模式 B: 无属性 (闲聊模式)
            else:
                instruction = "你是一个友好的北京旅游助手。请自然地与用户对话。"
                user_input = user_query

            # Qwen ChatML 格式构造
            # 注意：实际训练中我们通常只训练 assistant 的输出
            processed_data.append({
                "instruction": instruction,
                "input": user_input,
                "output": assistant_response
            })

    # 保存为 JSONL
    df = pd.DataFrame(processed_data)
    df.drop_duplicates(subset=['input'], inplace=True) # 简单去重
    df.to_json(output_file, orient='records', lines=True, force_ascii=False)
    print(f"✅ 数据处理完成！保存至 {output_file}，共 {len(df)} 条样本。")

if __name__ == "__main__":
    process_data("train.json", "train_data.jsonl")