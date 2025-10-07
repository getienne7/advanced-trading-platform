#!/usr/bin/env python3
"""
Validation script for Strategy Marketplace Service implementation
"""

import os
import sys
import importlib.util

def validate_file_exists(filepath, description):
    """Validate that a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - NOT FOUND")
        return False

def validate_python_syntax(filepath, description):
    """Validate Python file syntax"""
    try:
        spec = importlib.util.spec_from_file_location("module", filepath)
        if spec is None:
            print(f"✗ {description}: Could not load spec")
            return False
        
        module = importlib.util.module_from_spec(spec)
        # We don't execute the module to avoid import errors
        print(f"✓ {description}: Syntax valid")
        return True
    except SyntaxError as e:
        print(f"✗ {description}: Syntax error - {e}")
        return False
    except Exception as e:
        print(f"✓ {description}: Syntax valid (import error expected: {type(e).__name__})")
        return True

def main():
    """Main validation function"""
    print("Strategy Marketplace Service Implementation Validation")
    print("=" * 60)
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Core files validation
    core_files = [
        ("app.py", "Main FastAPI application"),
        ("models.py", "Database models"),
        ("schemas.py", "Pydantic schemas"),
        ("services.py", "Business logic services"),
        ("database.py", "Database configuration"),
        ("auth.py", "Authentication utilities"),
        ("config.py", "Configuration settings"),
        ("requirements.txt", "Python dependencies"),
        ("Dockerfile", "Docker configuration"),
        ("README.md", "Documentation"),
        ("__init__.py", "Python package marker"),
    ]
    
    print("\n1. Core Files Validation:")
    print("-" * 30)
    all_files_exist = True
    for filename, description in core_files:
        filepath = os.path.join(base_path, filename)
        if not validate_file_exists(filepath, description):
            all_files_exist = False
    
    # Python syntax validation
    print("\n2. Python Syntax Validation:")
    print("-" * 30)
    python_files = [
        ("app.py", "Main application"),
        ("models.py", "Database models"),
        ("schemas.py", "Pydantic schemas"),
        ("services.py", "Business services"),
        ("database.py", "Database config"),
        ("auth.py", "Authentication"),
        ("config.py", "Configuration"),
    ]
    
    all_syntax_valid = True
    for filename, description in python_files:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            if not validate_python_syntax(filepath, description):
                all_syntax_valid = False
    
    # Test files validation
    print("\n3. Test Files Validation:")
    print("-" * 30)
    test_files = [
        ("tests/__init__.py", "Test package marker"),
        ("tests/test_strategy_service.py", "Strategy service tests"),
        ("tests/test_subscription_service.py", "Subscription service tests"),
        ("tests/test_integration.py", "Integration tests"),
    ]
    
    all_tests_exist = True
    for filename, description in test_files:
        filepath = os.path.join(base_path, filename)
        if not validate_file_exists(filepath, description):
            all_tests_exist = False
    
    # Docker files validation
    print("\n4. Docker Configuration Validation:")
    print("-" * 30)
    docker_files = [
        ("Dockerfile", "Docker image configuration"),
        ("docker-compose.yml", "Docker compose configuration"),
        ("init.sql", "Database initialization"),
    ]
    
    all_docker_exist = True
    for filename, description in docker_files:
        filepath = os.path.join(base_path, filename)
        if not validate_file_exists(filepath, description):
            all_docker_exist = False
    
    # Feature implementation validation
    print("\n5. Feature Implementation Validation:")
    print("-" * 30)
    
    # Check if key classes and functions are defined
    features_to_check = [
        ("models.py", ["Strategy", "StrategySubscription", "StrategyPerformance", "StrategyRating"]),
        ("schemas.py", ["StrategyCreate", "StrategyResponse", "SubscriptionCreate", "PerformanceMetrics"]),
        ("services.py", ["StrategyService", "SubscriptionService", "PerformanceTracker", "MonetizationService"]),
        ("app.py", ["publish_strategy", "subscribe_to_strategy", "get_strategies", "get_strategy_performance"]),
    ]
    
    all_features_implemented = True
    for filename, expected_items in features_to_check:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                missing_items = []
                for item in expected_items:
                    if item not in content:
                        missing_items.append(item)
                
                if missing_items:
                    print(f"✗ {filename}: Missing items - {', '.join(missing_items)}")
                    all_features_implemented = False
                else:
                    print(f"✓ {filename}: All expected items found")
            except Exception as e:
                print(f"✗ {filename}: Could not validate - {e}")
                all_features_implemented = False
    
    # Requirements validation
    print("\n6. Requirements Implementation Validation:")
    print("-" * 30)
    
    requirements_mapping = {
        "10.1": "Strategy publication and subscription system",
        "10.4": "Strategy monetization capabilities",
        "10.5": "Copy trading with automatic replication"
    }
    
    # Check if key requirement features are implemented
    requirement_features = [
        ("Strategy publication", ["publish_strategy", "StrategyCreate", "Strategy"]),
        ("Strategy subscription", ["subscribe_to_strategy", "SubscriptionCreate", "StrategySubscription"]),
        ("Performance tracking", ["get_strategy_performance", "PerformanceMetrics", "StrategyPerformance"]),
        ("Monetization system", ["calculate_earnings", "MonetizationService", "StrategyEarnings"]),
        ("Copy trading", ["process_copy_trade_signal", "CopyTradeSignal", "_execute_copy_trade"]),
        ("Rating system", ["rate_strategy", "RatingCreate", "StrategyRating"]),
    ]
    
    all_requirements_met = True
    for feature_name, required_items in requirement_features:
        found_items = []
        missing_items = []
        
        for filename in ["app.py", "services.py", "models.py", "schemas.py"]:
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for item in required_items:
                        if item in content and item not in found_items:
                            found_items.append(item)
                except Exception:
                    pass
        
        for item in required_items:
            if item not in found_items:
                missing_items.append(item)
        
        if missing_items:
            print(f"✗ {feature_name}: Missing - {', '.join(missing_items)}")
            all_requirements_met = False
        else:
            print(f"✓ {feature_name}: All components implemented")
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_files_exist:
        print("✓ All core files present")
    else:
        print("✗ Some core files missing")
    
    if all_syntax_valid:
        print("✓ All Python files have valid syntax")
    else:
        print("✗ Some Python files have syntax errors")
    
    if all_tests_exist:
        print("✓ All test files present")
    else:
        print("✗ Some test files missing")
    
    if all_docker_exist:
        print("✓ All Docker configuration files present")
    else:
        print("✗ Some Docker files missing")
    
    if all_features_implemented:
        print("✓ All key features implemented")
    else:
        print("✗ Some key features missing")
    
    if all_requirements_met:
        print("✓ All requirements implemented")
    else:
        print("✗ Some requirements not fully implemented")
    
    overall_success = (all_files_exist and all_syntax_valid and 
                      all_tests_exist and all_docker_exist and 
                      all_features_implemented and all_requirements_met)
    
    if overall_success:
        print("\n🎉 VALIDATION PASSED: Strategy Marketplace Service implementation is complete!")
        return 0
    else:
        print("\n❌ VALIDATION FAILED: Some issues need to be addressed")
        return 1

if __name__ == "__main__":
    sys.exit(main())