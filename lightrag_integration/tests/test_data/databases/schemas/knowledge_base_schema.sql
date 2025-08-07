-- Knowledge Base Schema for LightRAG Testing
-- This schema supports document storage and retrieval operations

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT UNIQUE NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    metadata TEXT,  -- JSON formatted metadata
    document_type TEXT,
    source_path TEXT,
    file_hash TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processing_status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS document_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding_vector TEXT,  -- Serialized embedding vector
    embedding_model TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE TABLE IF NOT EXISTS knowledge_graph_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT UNIQUE NOT NULL,
    entity_type TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    properties TEXT,  -- JSON formatted properties
    document_sources TEXT,  -- JSON array of source document IDs
    confidence_score REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_graph_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relation_id TEXT UNIQUE NOT NULL,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    properties TEXT,  -- JSON formatted properties
    confidence_score REAL DEFAULT 0.0,
    document_sources TEXT,  -- JSON array of source document IDs
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_entity_id) REFERENCES knowledge_graph_entities(entity_id),
    FOREIGN KEY (target_entity_id) REFERENCES knowledge_graph_entities(entity_id)
);

CREATE TABLE IF NOT EXISTS query_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    query_type TEXT,
    response_text TEXT,
    retrieved_documents TEXT,  -- JSON array of document IDs
    processing_time_ms INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_embeddings_document ON document_embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON knowledge_graph_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_relations_source ON knowledge_graph_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON knowledge_graph_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_query_sessions_created ON query_sessions(created_at);

-- Insert sample test data
INSERT INTO documents (document_id, title, content, document_type, source_path, file_hash, processing_status) 
VALUES 
('doc_001', 'Sample Metabolomics Study', 'Clinical metabolomics analysis...', 'research_paper', '/test/sample1.pdf', 'abc123', 'completed'),
('doc_002', 'Clinical Trial Protocol', 'Phase II clinical trial...', 'clinical_trial', '/test/sample2.pdf', 'def456', 'completed'),
('doc_003', 'Biomarker Analysis', 'Identification of novel biomarkers...', 'research_paper', '/test/sample3.pdf', 'ghi789', 'processing');

INSERT INTO knowledge_graph_entities (entity_id, entity_type, entity_name, properties, document_sources, confidence_score)
VALUES
('ent_001', 'disease', 'Type 2 Diabetes', '{"synonyms": ["T2D", "diabetes mellitus type 2"]}', '["doc_001"]', 0.95),
('ent_002', 'metabolite', 'Glucose-6-phosphate', '{"formula": "C6H13O9P", "mass": 260.03}', '["doc_001"]', 0.90),
('ent_003', 'pathway', 'Glycolysis', '{"kegg_id": "map00010"}', '["doc_001"]', 0.88);

INSERT INTO knowledge_graph_relations (relation_id, source_entity_id, target_entity_id, relation_type, properties, confidence_score, document_sources)
VALUES
('rel_001', 'ent_001', 'ent_002', 'associated_with', '{"direction": "upregulated"}', 0.85, '["doc_001"]'),
('rel_002', 'ent_002', 'ent_003', 'participates_in', '{"role": "substrate"}', 0.92, '["doc_001"]');