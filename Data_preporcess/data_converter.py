import json
import re
from typing import List, Dict, Any

def convert_to_instruction_format(train_data: List[Dict]) -> List[Dict]:
    """
    将原始训练数据转换为{instruction, input, output}格式
    """
    formatted_data = []
    
    for dialog in train_data:
        messages = dialog["messages"]
        context = []
        
        for i, msg in enumerate(messages):
            if "message" in msg and msg["message"].strip():
                if i % 2 == 0:  # 用户输入
                    user_input = msg["message"].strip()
                    # 构建instruction
                    if context:
                        instruction = "请基于以下对话历史回答问题：\n" + "\n".join(context)
                    else:
                        instruction = "请回答以下问题："
                    
                    # 寻找对应的助手回复
                    if i + 1 < len(messages) and "message" in messages[i + 1]:
                        assistant_output = messages[i + 1]["message"].strip()
                        
                        formatted_data.append({
                            "instruction": instruction,
                            "input": user_input,
                            "output": assistant_output
                        })
                    
                    # 更新对话上下文
                    context.append(f"用户: {user_input}")
                    if i + 1 < len(messages) and "message" in messages[i + 1]:
                        context.append(f"助手: {assistant_output}")
    
    return formatted_data

# 加载原始数据
with open('./train.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# 转换格式
formatted_data = convert_to_instruction_format(original_data)

# 保存转换后的数据
with open('lora_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)