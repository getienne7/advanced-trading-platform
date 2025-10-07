-- Initialize Strategy Marketplace Database

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS strategy_marketplace;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create indexes for better performance
-- These will be created after tables are created by SQLAlchemy

-- Strategy indexes
-- CREATE INDEX IF NOT EXISTS idx_strategies_creator_id ON strategies(creator_id);
-- CREATE INDEX IF NOT EXISTS idx_strategies_category ON strategies(category);
-- CREATE INDEX IF NOT EXISTS idx_strategies_is_public ON strategies(is_public);
-- CREATE INDEX IF NOT EXISTS idx_strategies_is_active ON strategies(is_active);
-- CREATE INDEX IF NOT EXISTS idx_strategies_average_rating ON strategies(average_rating);
-- CREATE INDEX IF NOT EXISTS idx_strategies_created_at ON strategies(created_at);

-- Subscription indexes
-- CREATE INDEX IF NOT EXISTS idx_subscriptions_strategy_id ON strategy_subscriptions(strategy_id);
-- CREATE INDEX IF NOT EXISTS idx_subscriptions_subscriber_id ON strategy_subscriptions(subscriber_id);
-- CREATE INDEX IF NOT EXISTS idx_subscriptions_is_active ON strategy_subscriptions(is_active);
-- CREATE INDEX IF NOT EXISTS idx_subscriptions_subscribed_at ON strategy_subscriptions(subscribed_at);

-- Performance indexes
-- CREATE INDEX IF NOT EXISTS idx_performance_strategy_id ON strategy_performance(strategy_id);
-- CREATE INDEX IF NOT EXISTS idx_performance_period ON strategy_performance(period_start, period_end);
-- CREATE INDEX IF NOT EXISTS idx_performance_calculated_at ON strategy_performance(calculated_at);

-- Rating indexes
-- CREATE INDEX IF NOT EXISTS idx_ratings_strategy_id ON strategy_ratings(strategy_id);
-- CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON strategy_ratings(user_id);
-- CREATE INDEX IF NOT EXISTS idx_ratings_created_at ON strategy_ratings(created_at);

-- Earnings indexes
-- CREATE INDEX IF NOT EXISTS idx_earnings_strategy_id ON strategy_earnings(strategy_id);
-- CREATE INDEX IF NOT EXISTS idx_earnings_creator_id ON strategy_earnings(creator_id);
-- CREATE INDEX IF NOT EXISTS idx_earnings_period ON strategy_earnings(period_start, period_end);

-- Leaderboard indexes
-- CREATE INDEX IF NOT EXISTS idx_leaderboard_strategy_id ON strategy_leaderboard(strategy_id);
-- CREATE INDEX IF NOT EXISTS idx_leaderboard_period ON strategy_leaderboard(period, period_start, period_end);
-- CREATE INDEX IF NOT EXISTS idx_leaderboard_overall_rank ON strategy_leaderboard(overall_rank);

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE strategy_marketplace TO postgres;