-- Initialize trading platform database
-- This script runs when PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, executed_at);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id, executed_at);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol, exchange);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_id);

-- Create views for common queries
CREATE OR REPLACE VIEW v_active_positions AS
SELECT 
    p.*,
    s.name as strategy_name,
    s.description as strategy_description
FROM positions p
JOIN strategies s ON p.strategy_id = s.id
WHERE p.quantity != 0;

CREATE OR REPLACE VIEW v_daily_pnl AS
SELECT 
    DATE(executed_at) as trade_date,
    strategy_id,
    symbol,
    exchange,
    SUM(CASE WHEN side = 'BUY' THEN -quantity * price ELSE quantity * price END) as realized_pnl,
    COUNT(*) as trade_count
FROM trades
GROUP BY DATE(executed_at), strategy_id, symbol, exchange
ORDER BY trade_date DESC;

-- Create function for updating position timestamps
CREATE OR REPLACE FUNCTION update_position_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for position updates
DROP TRIGGER IF EXISTS trigger_update_position_timestamp ON positions;
CREATE TRIGGER trigger_update_position_timestamp
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_position_timestamp();

-- Insert default strategies
INSERT INTO strategies (id, name, description, parameters, is_active) VALUES
('default-arbitrage', 'Default Arbitrage Strategy', 'Basic arbitrage strategy for price differences', '{"min_profit": 0.001, "max_position_size": 1000}', true),
('default-momentum', 'Default Momentum Strategy', 'Basic momentum trading strategy', '{"lookback_period": 20, "threshold": 0.02}', true)
ON CONFLICT (id) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;