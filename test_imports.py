#!/usr/bin/env python3
"""
Test script to check if all imports work correctly.
"""
import sys
from pathlib import Path

# Add the shared directory to the path
sys.path.append(str(Path(__file__).parent / "shared"))

try:
    print("Testing imports...")
    
    # Test basic imports
    import asyncio
    import os
    from typing import Dict, List, Optional, Any, Union
    from datetime import datetime
    from decimal import Decimal
    print("✓ Basic imports successful")
    
    # Test database-specific imports
    try:
        import asyncpg
        print("✓ asyncpg import successful")
    except ImportError as e:
        print(f"✗ asyncpg import failed: {e}")
    
    try:
        import aioredis
        print("✓ aioredis import successful")
    except ImportError as e:
        print(f"✗ aioredis import failed: {e}")
    
    try:
        from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
        from influxdb_client import Point
        print("✓ InfluxDB client import successful")
    except ImportError as e:
        print(f"✗ InfluxDB client import failed: {e}")
    
    try:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
        from sqlalchemy.orm import declarative_base
        from sqlalchemy import Column, Integer, String, DateTime, Numeric, Boolean, Text, ForeignKey
        print("✓ SQLAlchemy imports successful")
    except ImportError as e:
        print(f"✗ SQLAlchemy imports failed: {e}")
    
    try:
        import structlog
        print("✓ structlog import successful")
    except ImportError as e:
        print(f"✗ structlog import failed: {e}")
    
    print("\nAll import tests completed!")
    
except Exception as e:
    print(f"Import test failed: {e}")
    sys.exit(1)