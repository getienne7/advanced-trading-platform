"""
WebSocket Manager - Real-time analytics data streaming
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from fastapi import WebSocket
import weakref

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time analytics streaming
    """
    
    def __init__(self):
        # Use WeakSet to automatically clean up closed connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.user_subscriptions: Dict[str, Set[str]] = {}
        self.running = False
        self.broadcast_interval = 1  # seconds
        
    async def add_client(self, user_id: str, websocket: WebSocket):
        """Add a new WebSocket client"""
        try:
            if user_id not in self.active_connections:
                self.active_connections[user_id] = set()
                self.user_subscriptions[user_id] = set()
            
            self.active_connections[user_id].add(websocket)
            
            # Send welcome message
            await self._send_to_client(websocket, {
                'type': 'connection_established',
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info(f"Added WebSocket client for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding WebSocket client: {e}")
    
    async def remove_client(self, user_id: str, websocket: Optional[WebSocket] = None):
        """Remove a WebSocket client"""
        try:
            if user_id in self.active_connections:
                if websocket:
                    self.active_connections[user_id].discard(websocket)
                    if not self.active_connections[user_id]:
                        del self.active_connections[user_id]
                        if user_id in self.user_subscriptions:
                            del self.user_subscriptions[user_id]
                else:
                    # Remove all connections for user
                    del self.active_connections[user_id]
                    if user_id in self.user_subscriptions:
                        del self.user_subscriptions[user_id]
            
            logger.info(f"Removed WebSocket client for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error removing WebSocket client: {e}")
    
    async def handle_client_message(self, user_id: str, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                await self._handle_subscription(user_id, data)
            elif message_type == 'unsubscribe':
                await self._handle_unsubscription(user_id, data)
            elif message_type == 'ping':
                await self._handle_ping(user_id)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from user {user_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def broadcast_analytics_update(self, user_id: str, analytics_data: Dict[str, Any]):
        """Broadcast analytics update to user's connections"""
        try:
            if user_id not in self.active_connections:
                return
            
            message = {
                'type': 'analytics_update',
                'data': analytics_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Send to all connections for this user
            connections_to_remove = []
            for websocket in self.active_connections[user_id].copy():
                try:
                    await self._send_to_client(websocket, message)
                except Exception as e:
                    logger.error(f"Error sending to WebSocket: {e}")
                    connections_to_remove.append(websocket)
            
            # Clean up failed connections
            for websocket in connections_to_remove:
                self.active_connections[user_id].discard(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                if user_id in self.user_subscriptions:
                    del self.user_subscriptions[user_id]
                    
        except Exception as e:
            logger.error(f"Error broadcasting analytics update: {e}")
    
    async def broadcast_risk_alert(self, user_id: str, alert_data: Dict[str, Any]):
        """Broadcast risk alert to user's connections"""
        try:
            if user_id not in self.active_connections:
                return
            
            message = {
                'type': 'risk_alert',
                'data': alert_data,
                'priority': alert_data.get('priority', 'medium'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self._broadcast_to_user(user_id, message)
            
        except Exception as e:
            logger.error(f"Error broadcasting risk alert: {e}")
    
    async def broadcast_performance_update(self, user_id: str, performance_data: Dict[str, Any]):
        """Broadcast performance update to user's connections"""
        try:
            if user_id not in self.active_connections:
                return
            
            message = {
                'type': 'performance_update',
                'data': performance_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self._broadcast_to_user(user_id, message)
            
        except Exception as e:
            logger.error(f"Error broadcasting performance update: {e}")
    
    async def start_broadcast_loop(self):
        """Start the main broadcast loop"""
        self.running = True
        logger.info("Starting WebSocket broadcast loop")
        
        while self.running:
            try:
                # Send periodic updates to all connected users
                for user_id in list(self.active_connections.keys()):
                    await self._send_periodic_update(user_id)
                
                await asyncio.sleep(self.broadcast_interval)
                
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(self.broadcast_interval)
    
    async def stop(self):
        """Stop the WebSocket manager"""
        self.running = False
        
        # Close all connections
        for user_id, connections in self.active_connections.items():
            for websocket in connections.copy():
                try:
                    await websocket.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
        
        self.active_connections.clear()
        self.user_subscriptions.clear()
        
        logger.info("WebSocket manager stopped")
    
    async def _handle_subscription(self, user_id: str, data: Dict[str, Any]):
        """Handle subscription request"""
        try:
            channels = data.get('channels', [])
            
            if user_id not in self.user_subscriptions:
                self.user_subscriptions[user_id] = set()
            
            for channel in channels:
                self.user_subscriptions[user_id].add(channel)
            
            # Send confirmation
            await self._broadcast_to_user(user_id, {
                'type': 'subscription_confirmed',
                'channels': list(self.user_subscriptions[user_id]),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info(f"User {user_id} subscribed to channels: {channels}")
            
        except Exception as e:
            logger.error(f"Error handling subscription: {e}")
    
    async def _handle_unsubscription(self, user_id: str, data: Dict[str, Any]):
        """Handle unsubscription request"""
        try:
            channels = data.get('channels', [])
            
            if user_id in self.user_subscriptions:
                for channel in channels:
                    self.user_subscriptions[user_id].discard(channel)
            
            # Send confirmation
            await self._broadcast_to_user(user_id, {
                'type': 'unsubscription_confirmed',
                'channels': channels,
                'remaining_channels': list(self.user_subscriptions.get(user_id, [])),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info(f"User {user_id} unsubscribed from channels: {channels}")
            
        except Exception as e:
            logger.error(f"Error handling unsubscription: {e}")
    
    async def _handle_ping(self, user_id: str):
        """Handle ping message"""
        try:
            await self._broadcast_to_user(user_id, {
                'type': 'pong',
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling ping: {e}")
    
    async def _send_periodic_update(self, user_id: str):
        """Send periodic update to user"""
        try:
            if user_id not in self.user_subscriptions:
                return
            
            subscriptions = self.user_subscriptions[user_id]
            
            # Only send updates if user is subscribed to relevant channels
            if 'pnl' in subscriptions or 'performance' in subscriptions or 'risk' in subscriptions:
                # Mock periodic data - in real implementation, would get from analytics engine
                periodic_data = {
                    'pnl': {
                        'total_pnl': 1500.0,
                        'unrealized_pnl': 1000.0,
                        'realized_pnl': 500.0
                    } if 'pnl' in subscriptions else None,
                    'performance': {
                        'total_return': 0.15,
                        'sharpe_ratio': 1.2,
                        'win_rate': 0.65
                    } if 'performance' in subscriptions else None,
                    'risk': {
                        'var_95': 2500.0,
                        'risk_score': 6.5,
                        'max_drawdown': -800.0
                    } if 'risk' in subscriptions else None
                }
                
                # Filter out None values
                filtered_data = {k: v for k, v in periodic_data.items() if v is not None}
                
                if filtered_data:
                    await self.broadcast_analytics_update(user_id, filtered_data)
                    
        except Exception as e:
            logger.error(f"Error sending periodic update: {e}")
    
    async def _broadcast_to_user(self, user_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections for a user"""
        try:
            if user_id not in self.active_connections:
                return
            
            connections_to_remove = []
            for websocket in self.active_connections[user_id].copy():
                try:
                    await self._send_to_client(websocket, message)
                except Exception as e:
                    logger.error(f"Error sending to WebSocket: {e}")
                    connections_to_remove.append(websocket)
            
            # Clean up failed connections
            for websocket in connections_to_remove:
                self.active_connections[user_id].discard(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                if user_id in self.user_subscriptions:
                    del self.user_subscriptions[user_id]
                    
        except Exception as e:
            logger.error(f"Error broadcasting to user: {e}")
    
    async def _send_to_client(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to a specific WebSocket client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to WebSocket client: {e}")
            raise
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())
    
    def get_user_connection_count(self, user_id: str) -> int:
        """Get number of connections for a specific user"""
        return len(self.active_connections.get(user_id, set()))
    
    def get_connected_users(self) -> List[str]:
        """Get list of users with active connections"""
        return list(self.active_connections.keys())