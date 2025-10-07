#!/usr/bin/env python3
"""
Simple GitHub Setup Script for Advanced Trading Platform
"""

import os
from pathlib import Path

def create_basic_files():
    """Create essential GitHub files"""
    
    # Create .github directory structure
    github_dir = Path(".github")
    workflows_dir = github_dir / "workflows"
    issue_templates_dir = github_dir / "ISSUE_TEMPLATE"
    
    github_dir.mkdir(exist_ok=True)
    workflows_dir.mkdir(exist_ok=True)
    issue_templates_dir.mkdir(exist_ok=True)
    
    # Basic CI workflow
    ci_workflow = """name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest flake8
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Test with pytest
      run: |
        pytest
"""
    
    with open(workflows_dir / "ci.yml", "w", encoding="utf-8") as f:
        f.write(ci_workflow)
    
    # Bug report template
    bug_template = """---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'

---

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
 - OS: [e.g. Ubuntu 20.04]
 - Python version: [e.g. 3.11]
 - Platform version: [e.g. v1.0.0]

**Additional context**
Add any other context about the problem here.
"""
    
    with open(issue_templates_dir / "bug_report.md", "w", encoding="utf-8") as f:
        f.write(bug_template)
    
    # Feature request template
    feature_template = """---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: 'enhancement'

---

**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Additional context**
Add any other context about the feature request here.
"""
    
    with open(issue_templates_dir / "feature_request.md", "w", encoding="utf-8") as f:
        f.write(feature_template)
    
    # Pull request template
    pr_template = """## Description
Brief description of changes made in this PR.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
"""
    
    with open(github_dir / "pull_request_template.md", "w", encoding="utf-8") as f:
        f.write(pr_template)
    
    print("✅ GitHub templates created")

def create_license():
    """Create MIT license"""
    
    license_text = """MIT License

Copyright (c) 2024 Advanced Trading Platform

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open("LICENSE", "w", encoding="utf-8") as f:
        f.write(license_text)
    
    print("✅ MIT License created")

def create_env_example():
    """Create environment example file"""
    
    env_example = """# Advanced Trading Platform - Environment Configuration

# Database Configuration
DATABASE_URL=postgresql://trading_user:trading_password@localhost:5432/trading_platform
REDIS_URL=redis://localhost:6379/0
RABBITMQ_URL=amqp://trading_user:trading_password@localhost:5672

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Exchange API Keys (Use testnet/sandbox for development)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET_KEY=your_coinbase_secret_key
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET_KEY=your_kraken_secret_key

# AI/ML Configuration
OPENAI_API_KEY=your_openai_api_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key

# Monitoring
SENTRY_DSN=your_sentry_dsn
DATADOG_API_KEY=your_datadog_key

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Service URLs
API_GATEWAY_URL=http://localhost:8000
STRATEGY_MARKETPLACE_URL=http://localhost:8007
TRADING_ENGINE_URL=http://localhost:8002
"""
    
    with open(".env.example", "w", encoding="utf-8") as f:
        f.write(env_example)
    
    print("✅ Environment example created")

def main():
    """Main setup function"""
    
    print("Setting up Advanced Trading Platform for GitHub...")
    print("=" * 60)
    
    create_basic_files()
    create_license()
    create_env_example()
    
    print("\n" + "=" * 60)
    print("GitHub Setup Complete!")
    print("=" * 60)
    
    print("\nFiles created:")
    print("✅ .github/workflows/ci.yml - CI pipeline")
    print("✅ .github/ISSUE_TEMPLATE/ - Bug and feature templates")
    print("✅ .github/pull_request_template.md - PR template")
    print("✅ LICENSE - MIT License")
    print("✅ .env.example - Environment configuration template")
    
    print("\nNext steps:")
    print("1. Create GitHub repository")
    print("2. git remote add origin <your-repo-url>")
    print("3. git add .")
    print("4. git commit -m 'Initial commit: Advanced Trading Platform'")
    print("5. git push -u origin main")
    
    print("\nYour platform is ready for GitHub! 🚀")

if __name__ == "__main__":
    main()