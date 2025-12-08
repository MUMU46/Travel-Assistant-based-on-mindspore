# relation_router_build.py
import os
import json
import numpy as np
from tqdm import tqdm

# MindSpore + Transformers（用于推理embedding）
import mindspore as ms
from transformers import AutoTokenizer, AutoModel  # 注意不是 AutoModelForCausalLM

# 向量索引：优先 Faiss，若无则降级到 Annoy
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    from annoy import AnnoyIndex
    _HAS_FAISS = False

# ---------- 配置 ----------
TRAIN_JSON = "/mnt/data/train.json"   # 你上传的 train.json，本地路径
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 示例编码器（可替换）
EMB_DIM = 384  # 与模型输出维度一致（MiniLM v2 -> 384）
INDEX_FILE = "relation_index.faiss" if _HAS_FAISS else "relation_index.ann"
META_FILE = "relation_meta.json"
EMB_NPY = "relation_embeddings.npy"
TOPK = 3
# -----------------------

def extract_relations_from_trainjson(train_json_path=TRAIN_JSON):
    """
    从 train.json 中抽取所有 attrname（关系）与可选的 relation textual (attrname 或较长描述)。
    返回 dict: {relation_key: example_phrase}
    """
    relations = {}  # relation_key -> example textual phrase
    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        # item is an entity record: { "name": "...", "messages":[ ... ] }
        entity_name = item.get("name", "").strip()
        for msg in item.get("messages", []):
            for attr in msg.get("attrs", []) if isinstance(msg.get("attrs", []), list) else []:
                rname = attr.get("attrname", "").strip()
                rval = attr.get("attrvalue", "").strip()
                # Normalize relation key: use rname
                if not rname:
                    continue
                # use a textual description combining relation and example value to help embedding
                example_text = f"{rname}：{rval}"
                # prefer keeping a representative text (first seen)
                if rname not in relations:
                    relations[rname] = example_text
                else:
                    # if we have no value before, prefer a non-empty rval
                    if not relations[rname] and rval:
                        relations[rname] = example_text
    return relations

# ---------- Embedding helper ----------
class RelationEmbedder:
    def __init__(self, model_name=MODEL_NAME, device="CPU"):
        # MindSpore device config (按需修改)
        # 如果你在 Ascend 上，请在外部设置 ms.context.set_context(device_target="Ascend")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=False)
        # 使用 AutoModel（encoder）并在 MindSpore 下加载（若有 ms 后端支持）
        # 注意：部分 HF 模型可能未提供 MindSpore 权重，视安装环境而定
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=False, ms_dtype=ms.float32)
        # 如果需要将模型放到特定设备，请使用 mindspore.context.set_context(...) 预先设置
        # 我们不在这里硬编码 device_map

    def encode(self, texts, batch_size=32):
        """
        texts: list[str]
        returns np.ndarray shape (len(texts), dim)
        """
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="ms")
            # model returns BaseModelOutput with last_hidden_state
            out = self.model(**enc)
            last = out.last_hidden_state  # MindSpore Tensor: (B, L, D)
            # mean-pooling (masking)
            attention_mask = enc["attention_mask"]  # ms.Tensor (B, L)
            mask = attention_mask.expand_dims(-1).astype(ms.float32)  # (B, L, 1)
            summed = (last * mask).sum(axis=1)  # (B, D)
            denom = mask.sum(axis=1).clip(min=1e-9)
            mean_pooled = summed / denom  # (B, D) as ms.Tensor
            # 转 numpy
            emb_np = mean_pooled.asnumpy()
            # L2 normalize
            norm = np.linalg.norm(emb_np, axis=1, keepdims=True)
            emb_np = emb_np / np.clip(norm, 1e-10, None)
            all_embs.append(emb_np)
        return np.vstack(all_embs)


# ---------- 索引构建 ----------
def build_relation_index(relations: dict, embedder: RelationEmbedder, index_file=INDEX_FILE, meta_file=META_FILE, emb_npy=EMB_NPY):
    """
    relations: {relation_key: example_text}
    返回：保存 index 和 meta
    """
    relation_keys = list(relations.keys())
    relation_texts = [relations[k] for k in relation_keys]

    print(f"编码 {len(relation_texts)} 个 relation ...")
    embs = embedder.encode(relation_texts, batch_size=32)  # (N, D)
    np.save(emb_npy, embs)
    print(f"Embeddings shape: {embs.shape}, 已保存到 {emb_npy}")

    dim = embs.shape[1]
    if _HAS_FAISS:
        print("构建 Faiss 索引 ...")
        index = faiss.IndexFlatIP(dim)  # 使用内积（embedding 已归一化，相当于cosine）
        index.add(embs.astype(np.float32))
        faiss.write_index(index, index_file)
        print(f"Faiss 索引已保存：{index_file}")
    else:
        print("Faiss 未安装，使用 Annoy 作为替代 ...")
        t = AnnoyIndex(dim, 'angular')
        for i, v in enumerate(embs):
            t.add_item(i, v.astype(np.float32))
        t.build(10)
        t.save(index_file)
        print(f"Annoy 索引已保存：{index_file}")

    # 元数据：id -> relation_key, relation_text
    meta = {"keys": relation_keys, "texts": relation_texts, "model": MODEL_NAME, "dim": dim}
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Meta 已保存：{meta_file}")
    return index_file, meta_file, emb_npy

# ---------- 查询函数 ----------
class RelationRouter:
    def __init__(self, index_file=INDEX_FILE, meta_file=META_FILE, emb_npy=EMB_NPY):
        # load meta
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.keys = meta["keys"]
        self.texts = meta["texts"]
        self.dim = meta["dim"]

        # load index
        if _HAS_FAISS:
            self.index = faiss.read_index(index_file)
            self.use_faiss = True
        else:
            self.index = AnnoyIndex(self.dim, 'angular')
            self.index.load(index_file)
            self.use_faiss = False

        # build embedder for queries
        self.embedder = RelationEmbedder(model_name=MODEL_NAME)

    def route(self, query, topk=TOPK):
        q_emb = self.embedder.encode([query])[0].astype(np.float32)
        if self.use_faiss:
            # 返回 (D, )
            scores, ids = self.index.search(np.array([q_emb]), topk)
            ids = ids[0].tolist()
            scores = scores[0].tolist()
        else:
            ids = self.index.get_nns_by_vector(q_emb, topk, include_distances=False)
            # annoy 返回 ids
            scores = [None] * len(ids)
        results = []
        for idx, sc in zip(ids, scores):
            results.append({"relation": self.keys[idx], "example_text": self.texts[idx], "score": sc})
        return results

# ---------- 主流程 ----------
if __name__ == "__main__":
    print("抽取 relation ...")
    relations = extract_relations_from_trainjson(TRAIN_JSON)
    print(f"抽取到 {len(relations)} 个关系：示例 {list(relations.items())[:10]}")

    # 在调用 MindSpore 前设置设备（按需）
    # ms.context.set_context(device_target="CPU")   # 或 "Ascend"
    embedder = RelationEmbedder(model_name=MODEL_NAME)
    build_relation_index(relations, embedder)

    print("构建完成。示例查询：")
    router = RelationRouter()
    for q in ["门票多少钱", "需要买票吗", "周边有什么", "几点开放", "地址在哪"]:
        print(q, "->", router.route(q, topk=3))
