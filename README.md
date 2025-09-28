---
license: unknown
---

#### 项目配置

#### 输入以下命令安装MindSpore NLP：

```python
git clone https://github.com/mindspore-lab/mindnlp.git

cd mindnlp

bash scripts/build_and_reinstall.sh
```

#### 安装依赖和vllm-mindspore

```
git clone https://gitee.com/mindspore/vllm-mindspore.git
cd vllm-mindspore
git checkout master
git checkout 54d0598645abddd7b5cfd7ed9c6162d23c6752ad
```

#### 安装依赖

```
pip install numba setuptools_scm tokenizers==0.21.1
```

#### 用shell脚本安装vllm-mindspore

```
cd vllm-mindspore
bash install_depend_pkgs.sh
```

#### 输入以下命令，配置jupyter notebook

```python
pip install ipykernel

python -m ipykernel install --prefix=/home/ma-user/.local --name=py310 --display-name "Python 3.10"
```

#### 项目目录介绍

```

├── travalAssistant.ipynb #智能旅游助手问答效果测试与运行
├── Data_preporcess #数据预处理部分
│   ├── data_converter.py
│   ├── knowledge_extract.py
│   └── train.json #原数据集
├── README.md
├── TravelLoRA.ipynb #lora微调代码
├── Trip-1300.zip #微调后模型权重文件
├── knowledge_base.json #知识库文件
├── lora_training_data.json #用于lora微调的问答数据集文件
└── Assistant_vllm_ms.ipynb #vllm_mindspore推理框架运行问答助手
```

