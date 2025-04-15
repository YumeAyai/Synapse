import os
import re
from typing import List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
import logging
import numpy as np

# 加载环境变量
load_dotenv()
llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL"))


# Lucene查询生成模板
LUCENE_PROMPT_TEMPLATE = """
根据以下用户输入生成Lucene查询语句。请注意，以下是一些重要的规则和约束，务必严格遵守：

1. 关键词提取：
   - 识别用户需求核心概念（如"DNA修复"对应"dna repair"）
   - 关键词应为英语单词或词组，小写形式
2. 仅检索字段："name"和"definition"
3. 权重规则：
   - name字段固定权重^3.0
   - definition字段固定权重^1.5
4. 匹配规则：
   - 精确短语用双引号包裹
   - 自动添加模糊匹配~0.8（当用户提到"近义词"或"可能拼写错误"时）
5. 布尔逻辑：
   - 默认使用OR连接不同字段
   - 用户明确要求"同时包含"时使用AND
   - 出现否定词时使用NOT
6. 输出格式：
   - 纯文本Lucene查询
   - 禁止任何额外说明或格式
   - 输出示例格式：name:"关键词"^3.0 OR definition:"关键词"^1.5

当前用户输入：{input}
"""


# Neo4j数据库连接类
class Neo4jDatabase:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )

    def query(self, cypher_query, parameters=None):
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record for record in result]

    def close(self):
        self.driver.close()


# Cypher查询生成器
def create_cypher_chain():
    def cypher_generator(lucene_query=None, vector_query=None):
        if vector_query:
            # 向量检索
            cypher = """
            CALL db.index.vector.queryNodes("goTermVector", $top_k, $embedding)
            YIELD node, score
            RETURN node{ .id, .name, .definition, .namespace } AS node, score
            ORDER BY score DESC
            LIMIT $top_k
            """
            return cypher, {
                "top_k": vector_query["top_k"],
                "embedding": vector_query["embedding"],
            }
        else:
            # 文本检索
            cypher = """
            CALL db.index.fulltext.queryNodes("goTermTextIndex", $query)
            YIELD node, score
            RETURN node{ .id, .name, .definition, .namespace } AS node, score
            ORDER BY score DESC
            LIMIT 10
            """
            return cypher, {"query": lucene_query}

    return cypher_generator


# 图推理工具类
class GraphInferenceTool:
    def __init__(self, neo4j_db, cypher_generator):
        self.neo4j_db = neo4j_db
        self.cypher_generator = cypher_generator

    def execute(self, user_input, use_vector_search=False):
        if use_vector_search:
            # 向量检索
            embedding = generate_embedding(user_input)
            vector_query = {"top_k": 10, "embedding": embedding}
            cypher_query, parameters = self.cypher_generator(vector_query=vector_query)
        else:
            # 文本检索
            lucene_query = generate_lucene(user_input)
            cypher_query, parameters = self.cypher_generator(lucene_query=lucene_query)

        results = self.neo4j_db.query(cypher_query, parameters)
        return [
            {
                "id": record["node"]["id"],
                "name": record["node"]["name"],
                "definition": record["node"]["definition"],
                "namespace": record["node"]["namespace"],
                "matched_fields": record.get("matched_fields", []),
                "boosted_score": record.get("boosted_score", 0),
                "score": record["score"],
            }
            for record in results
        ]


# 初始化编码器
encoder = SentenceTransformer(os.getenv("EMBEDDING_MODEL"), device="cpu")


# 生成文本嵌入
def generate_embedding(input_text):
    embedding = encoder.encode(input_text, show_progress_bar=False)
    return embedding.tolist()


# 生成Lucene查询
def generate_lucene(input_text):
    prompt = PromptTemplate(template=LUCENE_PROMPT_TEMPLATE, input_variables=["input"])
    chain = prompt | llm | StrOutputParser()
    lucene_query = chain.invoke({"input": input_text})
    print(f"Generated Lucene Query: {lucene_query}")
    return lucene_query


# 生物知识工作流
class BioKnowledgeWorkflow:
    def __init__(self, neo4j_db):
        router_prompt = PromptTemplate(
            template="""
根据用户问题判断场景类型，不要解释直接输出编号：
1. 概念关系推导（例如：A和B有什么联系？X如何导致Y？）
2. 定义查询（例如：什么是X？解释Y的概念）
3. 路径解释（例如：如何从A到达B？完成X的步骤是什么？）
用户输入：{query}
        """,
            input_variables=["query"],
        )
        runnable_sequence = router_prompt | llm | StrOutputParser()

        def route(inputs: dict) -> dict:
            logging.info(f"Routing input: {inputs}")
            result = runnable_sequence.invoke(inputs)
            logging.info(f"Routing result: {result}")
            return result.strip()  # 去除多余空格

        self.router = RunnableLambda(route)

        self.relation_chain = RelationDerivationChain(neo4j_db)
        self.definition_chain = DefinitionRetrievalChain(neo4j_db)
        self.path_chain = PathExplanationChain(neo4j_db)

        self.routes = {
            "1": self.relation_chain,
            "2": self.definition_chain,
            "3": self.path_chain,
        }

    def run(self, query: str):
        try:
            inputs = {"query": query}
            result = self.router.invoke(inputs)
            chain_name = result

            if chain_name in self.routes:
                chain = self.routes[chain_name]
                logging.info(f"Selected chain: {chain_name}")
                return chain.run(query)
            else:
                logging.warning(f"Unknown chain name: {chain_name}")
                return f"未知场景编号：{chain_name}"
        except Exception as e:
            logging.error(f"Error during workflow execution: {str(e)}")
            return f"处理失败：{str(e)}"


class RelationDerivationChain(BaseModel):
    """概念关系推导链"""
    neo4j_db: dict = Field(default_factory=dict)
    start_node: dict = Field(default_factory=dict) 

    def __init__(self, neo4j_db: dict):
        super().__init__()
        self.neo4j_db = neo4j_db

    def _create_intent_chain(self):
        """意图解析：提取关键生物概念并生成英文术语列表"""
        prompt = PromptTemplate(
            template="""
分析以下用户输入，提取其中的关键生物概念，并将其转换为对应的英文术语列表：
输入：{query}
输出：一个包含英文术语的无序列表，每个术语用逗号分隔，且不包含任何其他文本或解释。
            """,
            input_variables=["query"],
        )
        return prompt | llm | CommaSeparatedListOutputParser()

    def _create_retrieval_chain(self):
        """模糊检索"""

        def retrieve(input_data):
            concept_list: List[str] = input_data["concepts"]
            results = []
            for concept in concept_list:
                concept = concept.strip()
                cypher_generator = create_cypher_chain()
                graph_inference_tool = GraphInferenceTool(
                    self.neo4j_db, cypher_generator
                )
                # 向量查询
                nodes = graph_inference_tool.execute(concept, use_vector_search=True)
                if not nodes:
                    logging.warning(f"未检索到与概念 '{concept}' 相关的节点")
                results.append(nodes)
            return results

        return retrieve

    def _find_shortest_paths(self):
        """图算法路径查找"""

        def find_paths(input_data):
            nodes = input_data["nodes"]
            if len(nodes) < 2:
                logging.warning("节点数量不足，无法查找路径")
                return {"paths": []}
            # 选择每个节点组中评分最高的节点
            for i in range(len(nodes)):
                nodes[i] = sorted(nodes[i], key=lambda x: x["score"], reverse=True)[:1]
            # 选择前两个节点
            nodes = [node[0] for node in nodes]

            start_id = nodes[0]["id"]
            end_id = nodes[1]["id"]
            cypher = """
            MATCH p = allShortestPaths((start:GO_Term)-[r*..5]-(end:GO_Term))
            WHERE start.id = $start_id AND end.id = $end_id
            RETURN 
                p AS path, 
                length(p) AS path_length,
                [rel IN relationships(p) | type(rel)] AS relationship_types
            ORDER BY path_length
            LIMIT 10
            """

            try:
                paths = self.neo4j_db.query(
                    cypher, {"start_id": start_id, "end_id": end_id}
                )
                if not paths:
                    logging.warning("未找到有效路径")
                    return {"paths": []}

                # 格式化路径结果
                formatted_paths = []
                for record in paths:
                    path = record["path"]
                    path_length = record["path_length"]
                    # 处理路径中的节点和关系
                    formatted_paths.append(
                        {
                            "length": path_length,
                            "relationship_types": [
                                {"type": rel.type, "properties": dict(rel)}
                                for rel in path.relationships
                            ],
                            "path": [
                                {
                                    "id": node["id"],
                                    "name": node["name"],
                                    "definition": node["definition"],
                                }
                                for node in path.nodes
                            ],
                        }
                    )
                return {"paths": formatted_paths}

            except Exception as e:
                logging.error(f"查询路径时发生错误：{str(e)}")
                return {"paths": []}

        return find_paths

    def _rank_paths(self):
        """路径排序"""
        def rank(input_data):
            paths = input_data["paths"]
            # 根据路径长度和节点权重排序
            if not paths:
                logging.warning("未找到有效路径")
                return {"ranked_paths": []}
            try:
                if paths is None:
                    self.start_node = paths[0]["path"][0]
                # 根据路径长度和节点权重排序
                sorted_paths = sorted(
                    paths, key=lambda p: (p["length"], -self._path_weight(p["path"]))
                )
                return {"ranked_paths": sorted_paths[:10]}
            except Exception as e:
                logging.error(f"路径排序时发生错误：{str(e)}")
                return {"ranked_paths": []}

        return rank

    def _path_weight(self, path):
        weight = 0
        for node in path:
            definition_length = len(node.get("definition", ""))
            embedding_similarity = self._calculate_embedding_similarity(node)
            weight += definition_length * embedding_similarity
        return weight

    def _calculate_embedding_similarity(self, node):
        start_embedding = self.start_node.get("embedding", [])
        current_embedding = node.get("embedding", [])
        if not start_embedding or not current_embedding:
            return 1.0  # 默认相似性
        similarity = np.dot(start_embedding, current_embedding) / (
            np.linalg.norm(start_embedding) * np.linalg.norm(current_embedding)
        )
        return max(0.0, similarity)  # 确保相似性非负

    def _generate_explanation(self):
        """路径解释生成"""
        prompt = PromptTemplate(
            template="""
            解释以下生物概念路径：
            {paths}
            要求用中文分点说明机制
            """,
            input_variables=["paths"],
        )
        return prompt | llm | StrOutputParser()

    def run(self, query: str):
        """执行概念关系推导"""
        try:
            # 意图解析
            intent_result = self._create_intent_chain().invoke({"query": query})
            logging.info(f"Intent Analysis Result: {intent_result}")

            # 混合检索
            retrieval_tool = self._create_retrieval_chain()
            nodes = retrieval_tool({"concepts": intent_result})
            logging.info(f"Retrieved Nodes: {nodes}")

            # 路径查找
            paths = self._find_shortest_paths()({"nodes": nodes})
            logging.info(f"Found Paths: {paths}")

            # 路径排序
            ranked_paths = self._rank_paths()(paths)
            logging.info(f"Ranked Paths: {ranked_paths}")

            # 路径解释
            explanation = self._generate_explanation().invoke(
                {"paths": ranked_paths["ranked_paths"]}
            )
            logging.info(f"Generated Explanation: {explanation}")

            return {"paths": ranked_paths["ranked_paths"], "explanation": explanation}
            # return f"以下是路径解释：\n{explanation}"
        except Exception as e:
            logging.error(f"Error during relation derivation: {str(e)}")
            return f"概念关系推导失败：{str(e)}"
        finally:
            self.neo4j_db.close()


class DefinitionRetrievalChain(BaseModel):
    """定义查询链"""

    neo4j_db: dict = Field(default_factory=dict)

    def __init__(self, neo4j_db: dict):
        super().__init__()
        self.neo4j_db = neo4j_db

    def run(self, query: str):
        """执行定义查询"""
        try:
            # 解析用户输入
            user_input = query.strip()
            cypher_generator = create_cypher_chain()
            graph_inference_tool = GraphInferenceTool(self.neo4j_db, cypher_generator)

            # 执行查询
            results = graph_inference_tool.execute(user_input, use_vector_search=False)

            # 构造上下文内容
            context = "\n".join(
                [
                    f"""
【{res['name']}】[{res['id']}]（https://www.ebi.ac.uk/QuickGO/term/{res['id']}）
匹配字段：{', '.join(res['matched_fields'])}
相关度：{res['score']:.2f}
定义：{res['definition']}
"""
                    for res in results
                ]
            )
            return f"以下是相关知识库内容：\n{context}"
        except Exception as e:
            logging.error(f"Error during definition retrieval: {str(e)}")
            return f"知识库查询失败：{str(e)}"
        finally:
            self.neo4j_db.close()

class PathExplanationChain(BaseModel):
    neo4j_db: dict = Field(default_factory=dict)

    def __init__(self, neo4j_db: dict):
        super().__init__()
        self.neo4j_db = neo4j_db

    def _extract_key_nodes(self, user_input):
        prompt = PromptTemplate(
            template="""
            提取以下输入中的关键生物节点，并转换为英文术语列表：
            输入：{query}
            输出：一个包含英文术语的无序列表，每个术语用逗号分隔，且不包含任何其他文本或解释。
            注意：确保提取的术语是有效的生物学术语。
            """,
            input_variables=["query"],
        )
        return prompt | llm | CommaSeparatedListOutputParser()

    def _find_paths_among_nodes(self, node_list):
        if len(node_list) < 2:
            logging.warning("节点数量不足，无法查找路径")
            return {"paths": []}

        # 构造 Cypher 查询：从多个节点出发找到包含所有目标节点的最短路径
        cypher = """
        MATCH (nodes:GO_Term)
        WHERE nodes.name IN $node_list
        WITH COLLECT(nodes) AS all_nodes
        UNWIND all_nodes AS start_node
        UNWIND all_nodes AS end_node
        WITH start_node, end_node
        WHERE start_node.id < end_node.id
        MATCH p = allShortestPaths((start_node)-[r*..5]-(end_node))
        RETURN 
            p AS path, 
            length(p) AS path_length,
            [rel IN relationships(p) | type(rel)] AS relationship_types
        ORDER BY path_length
        LIMIT 10
        """
        try:
            paths = self.neo4j_db.query(
                cypher, {"node_list": node_list}
            )
            if not paths:
                logging.warning("未找到有效路径")
                return {"paths": []}

            # 格式化路径结果
            formatted_paths = []
            for record in paths:
                path = record["path"]
                path_length = record["path_length"]
                formatted_paths.append(
                    {
                        "length": path_length,
                        "relationship_types": [
                            {"type": rel.type, "properties": dict(rel)}
                            for rel in path.relationships
                        ],
                        "path": [
                            {
                                "id": node["id"],
                                "name": node["name"],
                                "definition": node["definition"],
                            }
                            for node in path.nodes
                        ],
                    }
                )
            return {"paths": formatted_paths}
        except Exception as e:
            logging.error(f"查询路径时发生错误：{str(e)}")
            return {"paths": []}

    def _concatenate_paths(self, paths):
        """
        拼接多段路径并去重，同时保留节点和关系。
        """
        concatenated_path = []
        seen_nodes = set()
        seen_relationships = set()

        for path in paths:
            # 遍历路径中的每个节点和关系
            nodes = path["path"]
            relationships = path["relationship_types"]

            for i in range(len(nodes) - 1):
                # 当前节点和下一个节点
                current_node = nodes[i]
                next_node = nodes[i + 1]

                # 当前关系
                current_relationship = relationships[i]

                # 如果当前节点未见过，则添加到路径中
                if current_node["id"] not in seen_nodes:
                    concatenated_path.append({"type": "node", "data": current_node})
                    seen_nodes.add(current_node["id"])

                # 如果当前关系未见过，则添加到路径中
                relationship_key = f"{current_node['id']}->{next_node['id']}:{current_relationship['type']}"
                if relationship_key not in seen_relationships:
                    concatenated_path.append(
                        {
                            "type": "relationship",
                            "data": {
                                "source": current_node["id"],
                                "target": next_node["id"],
                                "type": current_relationship["type"],
                                "properties": current_relationship.get("properties", {}),
                            },
                        }
                    )
                    seen_relationships.add(relationship_key)

            # 添加最后一个节点（如果未见过）
            last_node = nodes[-1]
            if last_node["id"] not in seen_nodes:
                concatenated_path.append({"type": "node", "data": last_node})
                seen_nodes.add(last_node["id"])

        return concatenated_path

    def _generate_path_explanation(self):
        prompt = PromptTemplate(
            template="""
            使用中文解释以下生物通路的机制：
            {path}
            要求包含以下要素：
            1. 分子相互作用
            2. 调控关系
            3. 生物学意义
            """,
            input_variables=["path"],
        )
        return prompt | llm | StrOutputParser()

    def run(self, query: str):
        try:
            # 提取关键节点
            key_nodes = self._extract_key_nodes(query).invoke({"query": query})
            logging.info(f"Extracted Key Nodes: {key_nodes}")

            # 查找多节点路径
            paths = self._find_paths_among_nodes(key_nodes)
            logging.info(f"Found Paths: {paths}")

            # 拼接路径
            concatenated_path = self._concatenate_paths(paths["paths"])
            logging.info(f"Concatenated Path: {concatenated_path}")

            # 生成解释
            explanation = self._generate_path_explanation().invoke(
                {
                    "path": concatenated_path
                }
            )
            logging.info(f"Generated Explanation: {explanation}")

            return {
                "paths": concatenated_path,
                "explanation": explanation,
            }

            # return f"以下是路径解释：\n{explanation}"
        except Exception as e:
            logging.error(f"Error during path explanation: {str(e)}")
            return f"路径解释失败：{str(e)}"
        finally:
            self.neo4j_db.close()

# 主要RAG上下文生成函数
# def get_rag_context(user_input, use_vector_search=False):
#     neo4j_db = Neo4jDatabase()
#     cypher_generator = create_cypher_chain()
#     query = GraphInferenceTool(neo4j_db, cypher_generator)

#     try:
#         results = query.execute(user_input, use_vector_search=use_vector_search)
#         context = "\n".join(
#             [
#                 f"""
# 【{res['name']}】[{res['id']}]（https://www.ebi.ac.uk/QuickGO/term/{res['id']}）
# 匹配字段：{', '.join(res['matched_fields'])}
# 相关度：{res['score']:.2f}
# 定义：{res['definition']}
# """
#                 for res in results
#             ]
#         )
#         return f"以下是相关知识库内容：\n{context}"
#     except Exception as e:
#         return f"知识库查询失败：{str(e)}"
#     finally:
#         neo4j_db.close()

def get_rag_context(user_input):
    neo4j_db = Neo4jDatabase()
    workflow = BioKnowledgeWorkflow(neo4j_db)
    return workflow.run(user_input)

if __name__ == "__main__":
    neo4j_db = Neo4jDatabase()
    workflow = BioKnowledgeWorkflow(neo4j_db)

    print(workflow.run("什么是DNA复制？"))
    print(workflow.run("DNA和RNA有什么关系？"))
    print(workflow.run("如何从基因转录到蛋白质合成？"))
