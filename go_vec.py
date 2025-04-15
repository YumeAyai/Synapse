from sentence_transformers import SentenceTransformer
from pronto import Ontology
from neo4j import GraphDatabase
import logging
from tqdm import tqdm
import os
import dotenv

# 加载环境变量
dotenv.load_dotenv()

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GO-Importer")

class GOImporter:
    def __init__(self, neo4j_uri, neo4j_auth, 
                model_path=os.getenv("EMBEDDING_MODEL")):
        """
        增强初始化方法，包含：
        - 模型文件完整性校验
        - Neo4j连接测试
        - 内存优化
        """
        # 1. 模型校验
        # self._validate_model_files(model_path)
        
        # 2. 初始化编码器
        try:
            self.encoder = SentenceTransformer(model_path, device='cpu')
            # 测试编码器
            test_emb = self.encoder.encode("test")
            if len(test_emb) != 384:
                raise ValueError("模型维度异常，预期384维")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise

        # 3. Neo4j连接测试
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
            with self.driver.session() as session:
                session.run("RETURN 1 AS test").single()
        except Exception as e:
            logger.error(f"Neo4j连接失败: {str(e)}")
            self.driver = None
            raise

        # 4. 内存优化
        self.batch_size = 1000  # 根据内存调整
        self.stats = {
            'nodes_created': 0,
            'relationships_created': 0,
            'batches_processed': 0
        }

    def _preprocess_text(self, term):
        """构建语义化文本"""
        components = []
        if term.namespace:
            components.append(f"[{term.namespace.upper()}]")
        components.append(term.name)
        if term.definition:
            components.append(term.definition.text if hasattr(term.definition, 'text') else str(term.definition))
        return " :: ".join(components).replace('"', "'")

    def _process_batch(self, tx, term_batch, rel_batch):
        """处理单个批次的事务"""
        # 1. 导入节点
        node_result = tx.run("""
            UNWIND $terms AS term
            MERGE (t:GO_Term {id: term.id})
            SET t.name = term.name,
                t.namespace = term.namespace,
                t.definition = term.definition,
                t.embedding = term.embedding,
                t.text = term.text
            RETURN count(t) AS nodes_created
        """, terms=term_batch)
        self.stats['nodes_created'] += node_result.single()['nodes_created']

        # 2. 导入关系
        if rel_batch:
            rel_type_map = {}
            for rel_type, src, tgt in rel_batch:
                rel_type_map.setdefault(rel_type, []).append((src, tgt))

            for rel_type, pairs in rel_type_map.items():
                rel_result = tx.run(f"""
                    UNWIND $pairs AS pair
                    MATCH (src:GO_Term {{id: pair[0]}})
                    MATCH (tgt:GO_Term {{id: pair[1]}})
                    MERGE (src)-[:`{rel_type}`]->(tgt)
                    RETURN count(*) AS rels_created
                """, pairs=pairs)
                self.stats['relationships_created'] += rel_result.single()['rels_created']

    def import_ontology(self, obo_path, batch_size=500):
        """主导入流程"""
        go = Ontology(obo_path)
        
        with self.driver.session() as session:
            # 创建索引
            session.run("CREATE INDEX go_term_id IF NOT EXISTS FOR (t:GO_Term) ON (t.id)")
            session.run("""
                CREATE FULLTEXT INDEX goTermText IF NOT EXISTS 
                FOR (n:GO_Term) ON EACH [n.name, n.definition]
            """)
            session.run("""
                CREATE VECTOR INDEX goTermVector IF NOT EXISTS
                FOR (n:GO_Term) ON (n.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)

            term_batch = []
            rel_batch = []
            texts = []
            
            # 使用进度条
            total_terms = sum(1 for _ in go.terms() if not getattr(_, "is_obsolete", False))
            progress = tqdm(go.terms(), total=total_terms, desc="Processing Terms")

            for term in progress:
                if term.id is None or getattr(term, "is_obsolete", False):
                    continue

                # 处理节点
                text = self._preprocess_text(term)
                texts.append(text)
                term_data = {
                    "id": term.id,
                    "name": term.name,
                    "namespace": term.namespace,
                    "definition": str(term.definition) if term.definition else "",
                    "text": text
                }
                term_batch.append(term_data)

                # 处理关系
                for parent in term.superclasses():
                    if parent.id:
                    # and parent.id.startswith("GO:"):
                        rel_batch.append(("IS_A", term.id, parent.id))
                for rel in term.relationships:
                    for target in term.relationships[rel]:
                        if target.id:
                            # and target.id.startswith("GO:"):
                            rel_batch.append((rel.id.upper(), term.id, target.id))

                # 批量处理
                if len(term_batch) >= batch_size:
                    # 生成向量
                    embeddings = self.encoder.encode(texts, batch_size=128, show_progress_bar=False)
                    for i, term_data in enumerate(term_batch):
                        term_data["embedding"] = embeddings[i].tolist()
                    
                    # 提交事务
                    session.execute_write(self._process_batch, term_batch, rel_batch)
                    self.stats['batches_processed'] += 1
                    
                    # 重置批次
                    term_batch.clear()
                    rel_batch.clear()
                    texts.clear()

            # 处理剩余数据
            if term_batch:
                embeddings = self.encoder.encode(texts, batch_size=128, show_progress_bar=False)
                for i, term_data in enumerate(term_batch):
                    term_data["embedding"] = embeddings[i].tolist()
                session.execute_write(self._process_batch, term_batch, rel_batch)
                self.stats['batches_processed'] += 1

            # 打印统计信息
            logger.info(f"""
                Import Completed!
                Total Nodes: {self.stats['nodes_created']}
                Total Relationships: {self.stats['relationships_created']}
                Batches Processed: {self.stats['batches_processed']}
            """)

    def validate_import(self):
        """数据完整性校验"""
        with self.driver.session() as session:
            # 检查节点数量
            result = session.run("MATCH (t:GO_Term) RETURN count(t) AS count")
            node_count = result.single()["count"]
            
            # 检查向量完整性
            result = session.run("""
                MATCH (t:GO_Term) 
                WHERE t.embedding IS NULL 
                RETURN count(t) AS missing_vectors
            """)
            missing_vectors = result.single()["missing_vectors"]
            
            logger.info(f"Validation Results:\n- Total Nodes: {node_count}\n- Missing Vectors: {missing_vectors}")

if __name__ == "__main__":
    # 使用示例
    importer = GOImporter(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_auth=(
            os.getenv("NEO4J_USERNAME"), 
            os.getenv("NEO4J_PASSWORD")
        )
    )
    
    try:
        importer.import_ontology("go.obo", batch_size=1000)
        importer.validate_import()
    except Exception as e:
        logger.error(f"Import failed: {str(e)}")
    finally:
        importer.driver.close()