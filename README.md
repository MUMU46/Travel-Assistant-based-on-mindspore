# Travel Assistant based on MindSpore

项目实现了基于 MindSpore 的 LoRA（低秩适配）语言风格微调，并结合 RAG（检索增强生成）构建一个面向旅游场景的对话助手。

**项目特色**

- **模型微调**: 使用 LoRA 方法对大模型进行高效微调，降低训练成本。
- **RAG 对话**: 将检索模块与生成模块结合，用知识库增强对话的事实性和专业性。
- **MindSpore 生态**: 基于 `mindspore` 与 `mindnlp`。

**仓库示例结构**

- `train_qwen.py`：LoRA 微调脚本。
- `main.py` / `graph_rag.py`：RAG 相关逻辑与对话主流程示例。
- `data_preporcess/`：数据预处理脚本与训练数据示例。
- `qwen2.5-7b_lora_output/`：示例的 LoRA 输出（adapter 权重、配置等）。
- `knowledge_graph_cache.pkl`: graph_rag.py生成的知识图谱缓存文件。

**环境与依赖**

- 推荐使用虚拟环境（`venv` / `conda`）。
- 本项目的 Python 依赖列于 `requirements.txt`。注意：`mindspore` 在不同平台和硬件（CPU/GPU/Ascend）下安装方式不同，如需加速请参考 MindSpore 官方安装说明。

典型安装（bash / WSL / Git Bash）:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell) 激活虚拟环境:

```
.venv\Scripts\Activate.ps1
```

如果需要特定平台的 MindSpore wheel，请参考 MindSpore 官方文档并使用对应的安装命令。

**快速开始**

1. 数据准备：在 `data_preporcess/` 中放置或生成训练数据（示例文件 `train.json`）。

   ```
   # input_file: 输入数据路径，默认为”./train.json”
   # output_file：预处理后数据保存路径，默认为”./train_data.jsonl”
   cd data_preprocess
   python data_pre.py  --input_file xxx --output_fil xxx
   ```

2. LoRA 微调：运行 `train_qwen.py`（或项目内对应的训练脚本），将 权重文件保存到输出文件夹路径（示例`qwen2.5-7b_lora_output/`）。
   - 示例：
   ```
   #model_path：模型名称或预训练权重路径
   # data_file：标注数据路径
   # output_dir：微调后LoRA Adapter权重保存路径
   python train_qwen.py --model_path xxx --data_file xxx --output_dir xxx
   ```

3. RAG 集成：运行`main.py` 启动检索 + 生成对话流程，加载 LoRA权重与知识库。

   ```
   #model_path : 使用的模型名路径
   #lora_path：lora微调权重文件路径
   #knowledge_file：知识库文件路径
   python main.py --model_path xxx --lora_path xxx --knowledge_file xxx
   ```

##### 运行效果

##### ![image-20251202202708705](https://s2.loli.net/2025/12/11/OkGiYBWCl6jfXab.png)

![image-20251202202748100](https://s2.loli.net/2025/12/11/bTY3iCFZ5ApBNft.png)

![image-20251230141839676](https://s2.loli.net/2025/12/30/btDIB74HjMxREAh.png)

**注意事项**

- MindSpore 与部分 PyPI 包版本在不同平台上兼容性不同，若遇到安装问题，请优先参考 MindSpore 官方安装页。

