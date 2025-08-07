-- Cost Tracking Database Schema for Testing
-- This schema matches the production cost tracking requirements

CREATE TABLE IF NOT EXISTS cost_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    model_name TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    session_id TEXT,
    query_hash TEXT,
    metadata TEXT,  -- JSON formatted metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS budget_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type TEXT NOT NULL,
    threshold_type TEXT NOT NULL,
    threshold_value REAL NOT NULL,
    current_value REAL NOT NULL,
    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active',
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS api_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    response_time_ms INTEGER,
    status_code INTEGER,
    request_size INTEGER,
    response_size INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    error_details TEXT
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_cost_tracking_timestamp ON cost_tracking(timestamp);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_operation ON cost_tracking(operation_type);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_session ON cost_tracking(session_id);
CREATE INDEX IF NOT EXISTS idx_budget_alerts_type ON budget_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint ON api_metrics(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_metrics_timestamp ON api_metrics(timestamp);

-- Insert sample test data
INSERT INTO cost_tracking (timestamp, operation_type, model_name, input_tokens, output_tokens, total_tokens, cost_usd, session_id, query_hash) 
VALUES 
('2024-08-07T10:00:00Z', 'query_processing', 'gpt-4', 500, 300, 800, 0.024, 'test_session_001', 'hash_001'),
('2024-08-07T10:05:00Z', 'pdf_processing', 'gpt-4', 1200, 800, 2000, 0.060, 'test_session_001', 'hash_002'),
('2024-08-07T10:10:00Z', 'embedding_generation', 'text-embedding-ada-002', 0, 0, 5000, 0.001, 'test_session_002', 'hash_003');

INSERT INTO budget_alerts (alert_type, threshold_type, threshold_value, current_value, status)
VALUES
('daily_limit_warning', 'cost', 10.00, 8.50, 'active'),
('token_usage_high', 'tokens', 50000, 45000, 'resolved'),
('api_rate_limit', 'requests_per_minute', 100, 95, 'active');

INSERT INTO api_metrics (endpoint, method, response_time_ms, status_code, request_size, response_size, session_id)
VALUES
('/api/query', 'POST', 1250, 200, 512, 2048, 'test_session_001'),
('/api/pdf/upload', 'POST', 5000, 200, 1048576, 256, 'test_session_001'),
('/api/health', 'GET', 50, 200, 0, 128, 'health_check_001');