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

#### 
