#!/usr/bin/env python3
"""
Deployment Validation Script for Advanced Trading Platform

This script validates that all services are running correctly after deployment.
"""

import requests
import time
import sys
from typing import Dict, List

class PlatformValidator:
    """Validates platform deployment and service health"""
    
    def __init__(self):
        self.services = {
            "API Gateway": "http://localhost:8000/health",
            "Strategy Marketplace": "http://localhost:8007/health", 
            "Web Dashboard": "http://localhost:8080",
            "Grafana": "http://localhost:3000",
            "Prometheus": "http://localhost:9090",
            "RabbitMQ": "http://localhost:15672"
        }
        
        self.results = {}
    
    def check_service(self, name: str, url: str) -> Dict[str, any]:
        """Check if a service is responding"""
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return {"status": "✅ HEALTHY", "response_time": response.elapsed.total_seconds()}
            else:
                return {"status": f"⚠️ UNHEALTHY (HTTP {response.status_code})", "response_time": None}
        except requests.exceptions.ConnectionError:
            return {"status": "❌ NOT RESPONDING", "response_time": None}
        except requests.exceptions.Timeout:
            return {"status": "⏰ TIMEOUT", "response_time": None}
        except Exception as e:
            return {"status": f"❌ ERROR: {str(e)}", "response_time": None}
    
    def validate_all_services(self):
        """Validate all platform services"""
        print("🚀 Advanced Trading Platform - Deployment Validation")
        print("=" * 60)
        
        print("\n🔍 Checking service health...")
        
        for service_name, service_url in self.services.items():
            print(f"\n📊 {service_name}:")
            print(f"   URL: {service_url}")
            
            result = self.check_service(service_name, service_url)
            self.results[service_name] = result
            
            print(f"   Status: {result['status']}")
            if result['response_time']:
                print(f"   Response Time: {result['response_time']:.3f}s")
    
    def check_api_endpoints(self):
        """Check specific API endpoints"""
        print(f"\n🔌 Testing API Endpoints...")
        
        api_tests = [
            ("API Gateway Health", "http://localhost:8000/health"),
            ("Strategy Marketplace Health", "http://localhost:8007/health"),
            ("API Documentation", "http://localhost:8000/docs"),
            ("Marketplace Documentation", "http://localhost:8007/docs")
        ]
        
        for test_name, url in api_tests:
            result = self.check_service(test_name, url)
            print(f"   {test_name}: {result['status']}")
    
    def generate_summary(self):
        """Generate validation summary"""
        print(f"\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        print("=" * 60)
        
        healthy_services = 0
        total_services = len(self.results)
        
        for service_name, result in self.results.items():
            status_icon = "✅" if "HEALTHY" in result['status'] else "❌"
            print(f"{status_icon} {service_name}: {result['status']}")
            
            if "HEALTHY" in result['status']:
                healthy_services += 1
        
        print(f"\n📊 Health Score: {healthy_services}/{total_services} services healthy")
        
        if healthy_services == total_services:
            print("🎉 ALL SERVICES HEALTHY - Platform ready for use!")
            return True
        elif healthy_services >= total_services * 0.7:
            print("⚠️ MOSTLY HEALTHY - Some services may need attention")
            return True
        else:
            print("❌ UNHEALTHY - Multiple services need attention")
            return False
    
    def provide_troubleshooting(self):
        """Provide troubleshooting guidance"""
        print(f"\n🔧 TROUBLESHOOTING GUIDE")
        print("-" * 40)
        
        unhealthy_services = [name for name, result in self.results.items() 
                            if "HEALTHY" not in result['status']]
        
        if unhealthy_services:
            print("❌ Unhealthy services detected:")
            for service in unhealthy_services:
                print(f"   • {service}")
            
            print(f"\n💡 Quick fixes to try:")
            print("1. Check if Docker containers are running:")
            print("   docker-compose ps")
            print("\n2. Restart unhealthy services:")
            print("   docker-compose restart")
            print("\n3. Check service logs:")
            print("   docker-compose logs -f [service-name]")
            print("\n4. Reset everything if needed:")
            print("   docker-compose down && docker-compose up -d")
        else:
            print("✅ All services are healthy!")
    
    def run_full_validation(self):
        """Run complete validation suite"""
        self.validate_all_services()
        self.check_api_endpoints()
        is_healthy = self.generate_summary()
        self.provide_troubleshooting()
        
        return is_healthy

def main():
    """Main validation function"""
    validator = PlatformValidator()
    
    print("⏳ Waiting 10 seconds for services to start...")
    time.sleep(10)
    
    is_healthy = validator.run_full_validation()
    
    if is_healthy:
        print(f"\n🚀 Platform is ready! Access points:")
        print("   🌐 Main Dashboard: http://localhost:8080")
        print("   🔌 API Gateway: http://localhost:8000")
        print("   🏪 Strategy Marketplace: http://localhost:8007")
        print("   📊 Monitoring: http://localhost:3000")
        
        print(f"\n📚 Next steps:")
        print("   1. Open the main dashboard in your browser")
        print("   2. Explore the strategy marketplace")
        print("   3. Check out the API documentation")
        print("   4. Join our Discord community for support")
        
        sys.exit(0)
    else:
        print(f"\n⚠️ Platform needs attention before use.")
        print("   Check the troubleshooting guide above.")
        sys.exit(1)

if __name__ == "__main__":
    main()