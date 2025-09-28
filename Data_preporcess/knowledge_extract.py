import json
import re
from typing import List, Dict, Any

def extract_knowledge_entries(train_data: List[Dict]) -> List[Dict]:
    """
    从原始数据中抽取知识条目用于RAG
    """
    knowledge_entries = []
    
    for dialog in train_data:
        scene_name = dialog.get("name", "")
        messages = dialog["messages"]
        
        for msg in messages:
            if "attrs" in msg:
                for attr in msg["attrs"]:
                    attrname = attr.get("attrname", "")
                    attrvalue = attr.get("attrvalue", "")
                    entity_name = attr.get("name", "")
                    
                    if attrvalue and entity_name:
                        # 创建知识条目
                        knowledge_entry = {
                            "id": f"{scene_name}_{attrname}_{entity_name}",
                            "content": f"{entity_name}的{attrname}是：{attrvalue}",
                            "metadata": {
                                "scene": scene_name,
                                "attribute": attrname,
                                "entity": entity_name
                            }
                        }
                        knowledge_entries.append(knowledge_entry)
            
            # 从Information属性中提取更详细的知识
            if "attrs" in msg:
                for attr in msg["attrs"]:
                    if attr.get("attrname") == "Information":
                        info_content = attr.get("attrvalue", "")
                        entity_name = attr.get("name", "")
                        
                        if info_content and entity_name:
                            # 分割信息内容为多个知识片段
                            segments = re.split(r'#\s+', info_content)
                            for segment in segments:
                                if segment.strip():
                                    knowledge_entry = {
                                        "id": f"{entity_name}_info_{hash(segment)}",
                                        "content": f"{entity_name}的相关信息：{segment.strip()}",
                                        "metadata": {
                                            "scene": scene_name,
                                            "entity": entity_name,
                                            "type": "information"
                                        }
                                    }
                                    knowledge_entries.append(knowledge_entry)
    
    return knowledge_entries

original_data = json.load(open('./train.json', 'r', encoding='utf-8'))
# 抽取知识条目
knowledge_base = extract_knowledge_entries(original_data)

# 保存知识库
with open('knowledge_base.json', 'w', encoding='utf-8') as f:
    json.dump(knowledge_base, f, ensure_ascii=False, indent=2)