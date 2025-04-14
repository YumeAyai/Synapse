from pronto import Ontology
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

def import_go_to_neo4j(obo_path):
    go = Ontology(obo_path)
    
    with driver.session() as session:
        # 创建索引加速查询
        session.run("CREATE INDEX IF NOT EXISTS FOR (t:GO_Term) ON (t.id)")
        
        # 批处理参数
        batch_size = 500
        term_batch = []
        rel_batch = []

        for term in go.terms():
            if not term.id or getattr(term, "is_obsolete", False):
                continue

            # 收集术语属性
            term_batch.append({
                "id": term.id,
                "name": term.name,
                "namespace": term.namespace,
                "definition": str(term.definition) if term.definition else ""
            })

            # 修复点1：调用superclasses()方法获取父类列表
            for parent in term.superclasses():  # 注意这里加了括号
                if parent.id:
                    rel_batch.append(("IS_A", term.id, parent.id))

            # 处理其他关系（如part_of等）
            for rel in term.relationships:
                for target in term.relationships[rel]:
                    if target.id:
                        rel_batch.append((rel.id.upper(), term.id, target.id))

            # 批量提交
            if len(term_batch) >= batch_size:
                _commit_batch(session, term_batch, rel_batch)
                term_batch.clear()
                rel_batch.clear()

        # 提交剩余数据
        if term_batch:
            _commit_batch(session, term_batch, rel_batch)

def _commit_batch(session, terms, relationships):
    # 批量创建节点
    session.run(
        """UNWIND $terms AS term
        MERGE (t:GO_Term {id: term.id})
        SET t.name = term.name,
            t.namespace = term.namespace,
            t.definition = term.definition
        """, 
        terms=terms
    )
    
    # 修复点2：动态处理关系类型
    if relationships:
        # 按关系类型分组批量处理
        rel_type_map = {}
        for rel in relationships:
            rel_type = rel[0]
            if rel_type not in rel_type_map:
                rel_type_map[rel_type] = []
            rel_type_map[rel_type].append( (rel[1], rel[2]) )

        # 为每种关系类型单独执行批量操作
        for rel_type, pairs in rel_type_map.items():
            session.run(
                f"""UNWIND $pairs AS pair
                MATCH (child:GO_Term {{id: pair[0]}})
                MATCH (parent:GO_Term {{id: pair[1]}})
                MERGE (child)-[:`{rel_type}`]->(parent)
                """,
                pairs=pairs
            )

if __name__ == "__main__":
    import_go_to_neo4j("go.obo")