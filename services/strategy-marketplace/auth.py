"""
Authentication utilities for Strategy Marketplace Service
"""

import jwt
import os
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
import httpx
from datetime import datetime

security = HTTPBearer()

# JWT configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Verify JWT token and return user information
    """
    token = credentials.credentials
    
    try:
        # First try to decode locally
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Check if token is expired
        if datetime.utcnow().timestamp() > payload.get("exp", 0):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        
        return payload
        
    except jwt.InvalidTokenError:
        # If local verification fails, try auth service
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{AUTH_SERVICE_URL}/verify-token",
                    headers={"Authorization": f"Bearer {token}"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token"
                    )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

async def get_current_user(token_data: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    """
    Get current user information from token
    """
    user_id = token_data.get("user_id") or token_data.get("sub")
    username = token_data.get("username") or token_data.get("name")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user_id"
        )
    
    return {
        "user_id": user_id,
        "username": username,
        "email": token_data.get("email"),
        "roles": token_data.get("roles", []),
        "permissions": token_data.get("permissions", [])
    }

def require_permission(permission: str):
    """
    Decorator to require specific permission
    """
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_permissions = current_user.get("permissions", [])
        if permission not in user_permissions and "admin" not in current_user.get("roles", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
        return current_user
    
    return permission_checker

def require_role(role: str):
    """
    Decorator to require specific role
    """
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_roles = current_user.get("roles", [])
        if role not in user_roles and "admin" not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role}"
            )
        return current_user
    
    return role_checker