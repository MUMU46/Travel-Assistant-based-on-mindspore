import re
from typing import List, Dict, Set, Tuple
import json
import networkx as nx
import logging
import pickle
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_CACHE_FILE = "knowledge_graph_cache.pkl"

# ------------------------------
# 关系词典（可扩展）
# ------------------------------
RELATION_KEYWORDS = {
    "门票": ["门票", "票价", "多少钱", "价格", "收费"],
    "开放时间": ["开放", "时间", "营业", "几点"],
    "地址": ["在哪", "位置", "地址", "怎么去"],
    "电话": ["电话", "联系方式", "咨询"],
    "周边景点": ["周边", "附近", "周围"],
    "Information": ["介绍", "信息", "简介"],
}

IMPORTANT_RELATIONS = ["Information", "门票", "开放时间", "地址", "电话", "周边景点"]

# ------------------------------
# 关系路由器
# ------------------------------
class RelationRouter:
    def __init__(self, entity_list, relation_keywords):
        self.entities = entity_list
        self.relation_keywords = relation_keywords

    def extract_entities(self, query: str):
        return [e for e in self.entities if e in query]

    def extract_relations(self, query: str):
        found = []
        for r, kws in self.relation_keywords.items():
            for kw in kws:
                if kw in query:
                    found.append(r)
                    break
        return found

    def route(self, query: str):
        ents = self.extract_entities(query)
        rels = self.extract_relations(query)

        if not ents:
            return {"mode": "no_entity", "entities": [], "relations": rels}

        if not rels:
            return {"mode": "entity_only", "entities": ents, "relations": []}

        return {"mode": "entity_relation", "entities": ents, "relations": rels}

# ------------------------------
# Graph RAG 实现（加入 Router）
# ------------------------------
class KnowledgeGraphRAG:
    def __init__(self, knowledge_file_path):
        self.graph = nx.Graph()
        self.entity_index = set()

        self.load_and_build_graph(knowledge_file_path)

        # 初始化 Router
        self.router = RelationRouter(
            entity_list=list(self.entity_index),
            relation_keywords=RELATION_KEYWORDS
        )

    def load_and_build_graph(self, file_path):
        if os.path.exists(GRAPH_CACHE_FILE):
            try:
                with open(GRAPH_CACHE_FILE, "rb") as f:
                    data = pickle.load(f)
                self.graph = data["graph"]
                self.entity_index = data["entity_index"]
                logger.info("已从缓存加载知识图谱")
                return
            except:
                pass

        logger.info(f"正在从 {file_path} 构建图谱...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            main_entity = item.get("name")
            if main_entity:
                self.graph.add_node(main_entity, type="entity")
                self.entity_index.add(main_entity)

            for msg in item.get("messages", []):
                for a in msg.get("attrs", []):
                    s, r, t = a.get("name"), a.get("attrname"), a.get("attrvalue")
                    if s and r and t:
                        self.graph.add_node(s, type="entity")
                        self.graph.add_node(t, type="value")
                        self.graph.add_edge(s, t, relation=r)
                        self.entity_index.add(s)

        with open(GRAPH_CACHE_FILE, "wb") as f:
            pickle.dump({"graph": self.graph, "entity_index": self.entity_index}, f)

        logger.info("图谱构建完成")

    # =====================================================
    # 核心：基于 Router 精准检索（替换掉你之前的 retrieve）
    # =====================================================
    def retrieve(self, query: str, max_facts=8):
        route = self.router.route(query)

        mode = route["mode"]
        entities = route["entities"]
        relations = route["relations"]

        # --------------------------
        # 1) 无实体 → 不检索
        # --------------------------
        if mode == "no_entity":
            return ""

        # --------------------------
        # 2) 实体 + 关系（最优）
        # --------------------------
        if mode == "entity_relation":
            results = []
            for e in entities:
                for nbr in self.graph.neighbors(e):
                    r = self.graph[e][nbr]["relation"]
                    if r in relations:
                        results.append(f"【{e}】的{r}是{nbr}")
            return "\n".join(f"【知识】: {x}。" for x in results[:max_facts])

        # --------------------------
        # 3) 只有实体 → 返回常用属性
        # --------------------------
        if mode == "entity_only":
            results = []
            for e in entities:
                for nbr in self.graph.neighbors(e):
                    r = self.graph[e][nbr]["relation"]
                    if r in IMPORTANT_RELATIONS:
                        results.append(f"【{e}】的{r}是{nbr}")
            return "\n".join(f"【知识】: {x}。" for x in results[:max_facts])

        return ""
