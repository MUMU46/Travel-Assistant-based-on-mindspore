import sys
import os
os.environ["HF_HOME"] = "/cache/hf"
os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
os.environ["HF_HUB_CACHE"] = "/cache/hf"
import argparse
import json
import mindspore as ms
import mindnlp
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import time
from graph_rag import KnowledgeGraphRAG

            
def should_retrieve(self, query, history):
    """
    使用 LLM 判断是否需要进行知识图谱检索。
    输入包括 query + 历史，使模型知道当前对话上下文。
    """
    
    # 构造可读的简短历史
    history_text = ""
    for q, a in history[-3:]:
        history_text += f"用户: {q}\n助手: {a}\n"

    decision_prompt = [
        {
            "role": "system",
            "content":
            "你是一个判断是否需要知识库检索的分类器。\n"
            "当用户问的问题与旅游景点、票价、地址、电话、开放时间、路线、附近、历史有关时 → 输出 YES。\n"
            "即使当前句子是模糊问句（如“还有吗？”、“那呢？”、“要”），只要根据”上下文“判断与之前旅游问题相关 → 输出 YES。\n"
            "若问题是闲聊、主观提问、自我介绍或无关语句 → 输出 NO。\n"
            "只输出 YES 或 NO，不要解释。\n"
        },
        {
            "role": "user",
            "content":
            f"【对话历史】\n{history_text}\n"
            f"【当前用户问题】\n{query}\n"
            "请判断是否需要知识库检索。"
        }
    ]

    # Tokenize
    inputs = self.tokenizer.apply_chat_template(
        decision_prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="ms"
    )

    # 移动到 NPU
    if 'Ascend' in ms.context.get_context('device_target'):
        inputs = {k: v.to("npu") for k, v in inputs.items()}

    output = self.model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False
    )

    text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip().upper()

    return "YES" in text


class GraphRAGServer:
    def __init__(self, model_path, lora_path, knowledge_path):
        """
        初始化 Graph RAG 服务器
        """
        self.knowledge_path = knowledge_path
        
        # 1. 初始化知识图谱
        self.kg = KnowledgeGraphRAG(knowledge_path)
        
        
        # 2. 加载模型和 Tokenizer
        print("正在加载模型...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                use_fast=False, 
                mirror='modelscope', 
                trust_remote_code=True
            )

            # MindSpore 环境下加载模型

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                ms_dtype=ms.bfloat16, # 或者 ms.bfloat16
                mirror='modelscope', 
                device_map=0,
                trust_remote_code=True
            )

            # 如果有 LoRA (PeftModel)，在这里加载
            if lora_path and os.path.exists(lora_path):
                from peft import PeftModel
                print(f"正在加载 LoRA 权重: {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, model_id=lora_path)

            print("模型加载完成！")

        except Exception as e:
            print(f"模型加载失败: {e}")
            

    def generate_response(self, query, history=[]):
        """
        生成回复流程
        """
    
        try:
            need_rag = should_retrieve(self.model, self.tokenizer, query, history)
        except:
            need_rag = True
            
        system_prompt = (
                "你是一名专业的北京旅游助手。\n"
                "你的任务是根据提供的知识或常识回答用户问题。\n"
                "不得编造事实。未在知识库中查询到答案时，礼貌地说明不知道。\n"
                "禁止输出思考过程。"
            )

         # ----------------- 构建 messages -----------------
        messages = [{"role": "system", "content": system_prompt}]

        # 历史对话
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        # Step 2: 如果需要检索
        if need_rag:
            # print("→ 本轮需要检索图谱")

            knowledge = self.kg.retrieve(query)
            # print('检索到的知识：',knowledge)
            
            if knowledge:
                # RAG 知识放在 “assistant” 角色，权重最高
                messages.append({
                    "role": "assistant",
                    "content": f"【知识库检索结果】\n{knowledge}"
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": "【知识库检索结果为空】"
                })

        else:
            print("→ 属于闲聊，不检索知识库")

        # 用户当前问题
        messages.append({"role": "user", "content": query})

      
        from transformers import TextIteratorStreamer # 确保这个导入语句在文件顶部
        
        # 1. 在 try 块外部定义 streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        # ----------------------------------------------------
        
        # Step 3: 模型推理
        try:
            # 1. Tokenizer生成输入
            input_ids_dict = self.tokenizer.apply_chat_template( # 重命名变量为 input_ids_dict
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="ms", # 返回 MindSpore Tensor
                return_dict=True
            )


            if 'Ascend' in ms.context.get_context('device_target'):
                # 遍历字典中的所有张量，并使用 .to('npu') 移动它们
                input_ids = {k: v.to('npu') for k, v in input_ids_dict.items()}
            else:
                # 如果不是 Ascend 环境，则保持原样或移动到 'cpu'
                input_ids = input_ids_dict
        
            
            generation_kwargs = dict(
                input_ids,
                streamer=streamer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            
            # 在独立线程中运行生成
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            generated_text = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                generated_text += new_text
            print() # 换行
            
            return generated_text
            
        except Exception as e:
            print(f"生成出错: {e}")
            return "抱歉，系统遇到了一些问题。"

def main():
    parser = argparse.ArgumentParser(description="Run Graph RAG Travel Assistant.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Path to the pre-trained model")
    parser.add_argument("--lora_path", type=str, default="./qwen2.5-7B_lora_output/checkpoint-800", help="Path to the LoRA checkpoint")
    parser.add_argument("--knowledge_file", type=str, default="train.json", help="Path to the knowledge base file")
    
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    LORA_PATH = args.lora_path
    KNOWLEDGE_FILE = args.knowledge_file
    
    if not os.path.exists(KNOWLEDGE_FILE):
        print(f"错误：找不到知识库文件 {KNOWLEDGE_FILE}")
        return

    server = GraphRAGServer(MODEL_PATH, LORA_PATH, KNOWLEDGE_FILE)
    
    history = []
    
    print("="*50)
    print("Graph RAG 旅游助手已启动 (输入 quit 退出)")
    print("="*50)
    
    while True:
        query = input("\n用户: ").strip()
        if query.lower() in ['quit', 'exit']:
            break
        if not query:
            continue
            
        print("助手: ", end="")
        # print("hisotry:",history)
        response = server.generate_response(query, history)
      
        # 更新简易历史
        history.append((query, response))
        if len(history) > 3:
            history.pop(0)

if __name__ == "__main__":
    main()