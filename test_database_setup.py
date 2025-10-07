#!/usr/bin/env python3
"""
Simple database setup test.
"""
import sys
from pathlib import Path

# Add the shared directory to the path
sys.path.append(str(Path(__file__).parent / "shared"))

def test_imports():
    """Test database imports."""
    print("Testing database imports...")
    
    try:
        from database import DatabaseConfig, DatabaseManager
        from database import PostgreSQLManager, RedisManager, InfluxDBManager
        from database import Trade, Position, Strategy, Base
        print("✓ All database imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test database configuration."""
    print("Testing database configuration...")
    
    try:
        from database import DatabaseConfig
        
        config = DatabaseConfig()
        print(f"✓ PostgreSQL URL: {config.postgres_url}")
        print(f"✓ Redis URL: {config.redis_url}")
        print(f"✓ InfluxDB URL: {config.influxdb_url}")
        print(f"✓ InfluxDB Org: {config.influxdb_org}")
        print(f"✓ InfluxDB Bucket: {config.influxdb_bucket}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_models():
    """Test database models."""
    print("Testing database models...")
    
    try:
        from database import Base, Trade, Position, Strategy
        
        # Check table names
        tables = Base.metadata.tables
        expected_tables = ['trades', 'positions', 'strategies']
        
        for table_name in expected_tables:
            if table_name in tables:
                print(f"✓ Table '{table_name}' defined")
            else:
                print(f"✗ Table '{table_name}' missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Models test failed: {e}")
        return False

def test_managers():
    """Test database managers."""
    print("Testing database managers...")
    
    try:
        from database import DatabaseManager, DatabaseConfig
        
        config = DatabaseConfig()
        db_manager = DatabaseManager()
        
        # Check that managers exist
        print(f"✓ PostgreSQL manager: {type(db_manager.postgres).__name__}")
        print(f"✓ Redis manager: {type(db_manager.redis).__name__}")
        print(f"✓ InfluxDB manager: {type(db_manager.influxdb).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Managers test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=== Database Infrastructure Validation ===\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Models", test_models),
        ("Managers", test_managers)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} Test ---")
        if test_func():
            print(f"✓ {test_name} test passed")
        else:
            print(f"✗ {test_name} test failed")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All database infrastructure tests passed!")
        print("\nNext steps:")
        print("1. Start services: docker-compose up -d postgres redis influxdb")
        print("2. Run setup: python scripts/setup_databases.py")
        print("3. Check health: python scripts/check_databases.py")
    else:
        print("❌ Some tests failed. Please fix the issues.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())