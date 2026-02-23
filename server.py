from fastapi import FastAPI, APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import asyncio
import socketio
from haversine import haversine, Unit
import math
import httpx

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'human_alert')]

# Socket.IO setup
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# Create the main app
app = FastAPI(title="Human Alert API", version="1.0.0")

# Create Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio, app)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

class DeviceRegister(BaseModel):
    device_id: str
    platform: str = "android"
    push_token: Optional[str] = None

class DeviceResponse(BaseModel):
    device_id: str
    registered_at: datetime
    is_banned: bool = False

class LocationUpdate(BaseModel):
    device_id: str
    latitude: float
    longitude: float
    heading: Optional[float] = None

class AlertTrigger(BaseModel):
    device_id: str
    latitude: float
    longitude: float
    heading: Optional[float] = None
    trigger_type: str = "button"  # button or power_button

class AlertResponse(BaseModel):
    alert_id: str
    device_id: str
    latitude: float
    longitude: float
    heading: Optional[float] = None
    timestamp: datetime
    status: str  # active, ended
    current_radius: int  # 300, 600, 1000 meters
    end_reason: Optional[str] = None

class AlertEnd(BaseModel):
    device_id: str
    alert_id: str

class RespondToAlert(BaseModel):
    device_id: str
    alert_id: str
    responder_latitude: float
    responder_longitude: float

class AdminBan(BaseModel):
    device_id: str
    reason: str

# ==================== CONSTANTS ====================
COOLDOWN_SECONDS = 60  # Cooldown between alerts
ALERT_DURATION_SECONDS = 1200  # 20 minutes auto-end (changed from 3 minutes)
RADIUS_STAGES = [300, 600, 1000]  # meters
RADIUS_EXPANSION_INTERVAL = 20  # seconds
ONLINE_TIMEOUT_SECONDS = 120  # Consider device online if seen within 2 minutes

# ==================== HELPER FUNCTIONS ====================

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters"""
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing from point 1 to point 2 in degrees"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360

async def get_devices_in_radius(latitude: float, longitude: float, radius: int, exclude_device: str = None) -> List[dict]:
    """Get all devices within a radius"""
    devices = await db.devices.find({
        "is_banned": {"$ne": True},
        "last_latitude": {"$exists": True},
        "last_longitude": {"$exists": True}
    }).to_list(10000)
    
    nearby_devices = []
    for device in devices:
        if device["device_id"] == exclude_device:
            continue
        if "last_latitude" not in device or "last_longitude" not in device:
            continue
        
        dist = calculate_distance(
            latitude, longitude,
            device["last_latitude"], device["last_longitude"]
        )
        
        if dist <= radius:
            device["distance"] = dist
            device["bearing"] = calculate_bearing(
                device["last_latitude"], device["last_longitude"],
                latitude, longitude
            )
            nearby_devices.append(device)
    
    return nearby_devices

async def send_push_notification(push_token: str, title: str, body: str, data: dict = None, is_emergency: bool = True):
    """Send push notification via Expo Push API with LOUD sound and vibration"""
    if not push_token:
        return
    
    message = {
        "to": push_token,
        "sound": "default",
        "title": title,
        "body": body,
        "priority": "high",
        "channelId": "emergency-alerts",
        "_displayInForeground": True,
        "badge": 1,
    }
    
    # Add Android-specific settings for maximum visibility and LOUD sound
    if is_emergency:
        message["android"] = {
            "channelId": "emergency-alerts",
            "priority": "max",
            "sound": True,
            "vibrate": [0, 500, 200, 500, 200, 500],
            "sticky": True,
        }
        message["ios"] = {
            "sound": True,
            "_displayInForeground": True,
            "interruptionLevel": "critical",  # Critical alerts bypass DND
        }
        # Ensure sound plays
        message["sound"] = "default"
        message["_contentAvailable"] = True
    
    if data:
        message["data"] = data
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "https://exp.host/--/api/v2/push/send",
                json=message,
                headers={"Content-Type": "application/json"}
            )
            logger.info(f"Push notification sent: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Push notification response: {response.text}")
    except Exception as e:
        logger.error(f"Failed to send push notification: {e}")

async def broadcast_alert_to_nearby(alert: dict, radius: int):
    """Send notifications to all devices in radius - OPTIMIZED FOR SPEED"""
    nearby_devices = await get_devices_in_radius(
        alert["latitude"], 
        alert["longitude"], 
        radius,
        exclude_device=alert["device_id"]
    )
    
    # Send all push notifications in PARALLEL for instant delivery
    push_tasks = []
    socket_tasks = []
    
    for device in nearby_devices:
        # Prepare push notification task
        if device.get("push_token"):
            push_tasks.append(
                send_push_notification(
                    device["push_token"],
                    "🚨 EMERGENCY ALERT",
                    "Help needed nearby! Someone needs assistance.",
                    {
                        "alert_id": alert["alert_id"],
                        "latitude": alert["latitude"],
                        "longitude": alert["longitude"],
                        "distance": device["distance"],
                        "type": "emergency_alert"
                    }
                )
            )
        
        # Prepare Socket.IO emission task
        socket_tasks.append(
            sio.emit('alert_received', {
                "alert_id": alert["alert_id"],
                "latitude": alert["latitude"],
                "longitude": alert["longitude"],
                "heading": alert.get("heading"),
                "distance": device["distance"],
                "bearing": device["bearing"],
                "timestamp": alert["timestamp"].isoformat(),
                "current_radius": radius
            }, room=device["device_id"])
        )
    
    # Execute ALL tasks concurrently for instant delivery
    if push_tasks:
        await asyncio.gather(*push_tasks, return_exceptions=True)
    if socket_tasks:
        await asyncio.gather(*socket_tasks, return_exceptions=True)
    
    # Update notification count
    await db.alerts.update_one(
        {"alert_id": alert["alert_id"]},
        {"$set": {"notified_count": len(nearby_devices)}}
    )
    
    return len(nearby_devices)

async def expand_radius_task(alert_id: str):
    """Background task to expand radius if no response"""
    for i, radius in enumerate(RADIUS_STAGES[1:], 1):
        await asyncio.sleep(RADIUS_EXPANSION_INTERVAL)
        
        # Check if alert is still active
        alert = await db.alerts.find_one({"alert_id": alert_id})
        if not alert or alert.get("status") != "active":
            return
        
        # Check if there are responders
        responders = await db.alert_responses.count_documents({"alert_id": alert_id})
        if responders > 0:
            logger.info(f"Alert {alert_id} has responders, stopping radius expansion")
            return
        
        # Expand radius
        await db.alerts.update_one(
            {"alert_id": alert_id},
            {"$set": {"current_radius": radius}}
        )
        
        # Broadcast to new devices
        notified = await broadcast_alert_to_nearby(alert, radius)
        logger.info(f"Alert {alert_id} expanded to {radius}m, notified {notified} devices")
        
        # Emit radius update via Socket.IO
        await sio.emit('radius_expanded', {
            "alert_id": alert_id,
            "new_radius": radius
        })

async def auto_end_alert_task(alert_id: str):
    """Background task to auto-end alert after duration"""
    await asyncio.sleep(ALERT_DURATION_SECONDS)
    
    alert = await db.alerts.find_one({"alert_id": alert_id})
    if alert and alert.get("status") == "active":
        await db.alerts.update_one(
            {"alert_id": alert_id},
            {
                "$set": {
                    "status": "ended",
                    "ended_at": datetime.utcnow(),
                    "end_reason": "auto_timeout"
                }
            }
        )
        await sio.emit('alert_ended', {
            "alert_id": alert_id,
            "reason": "auto_timeout"
        })
        logger.info(f"Alert {alert_id} auto-ended after {ALERT_DURATION_SECONDS} seconds")

# ==================== DEVICE ENDPOINTS ====================

@api_router.post("/device/register", response_model=DeviceResponse)
async def register_device(device: DeviceRegister):
    """Register a new device or update existing"""
    existing = await db.devices.find_one({"device_id": device.device_id})
    
    if existing:
        # Check if banned
        if existing.get("is_banned"):
            raise HTTPException(status_code=403, detail="Device is banned")
        
        # Update push token and last seen
        await db.devices.update_one(
            {"device_id": device.device_id},
            {
                "$set": {
                    "push_token": device.push_token,
                    "platform": device.platform,
                    "last_seen": datetime.utcnow()
                }
            }
        )
        return DeviceResponse(
            device_id=device.device_id,
            registered_at=existing["registered_at"],
            is_banned=False
        )
    
    # Create new device
    device_doc = {
        "device_id": device.device_id,
        "platform": device.platform,
        "push_token": device.push_token,
        "registered_at": datetime.utcnow(),
        "last_seen": datetime.utcnow(),
        "is_banned": False,
        "alert_count": 0
    }
    
    await db.devices.insert_one(device_doc)
    
    return DeviceResponse(
        device_id=device.device_id,
        registered_at=device_doc["registered_at"],
        is_banned=False
    )

@api_router.put("/device/location")
async def update_device_location(location: LocationUpdate):
    """Update device location"""
    result = await db.devices.update_one(
        {"device_id": location.device_id},
        {
            "$set": {
                "last_latitude": location.latitude,
                "last_longitude": location.longitude,
                "last_heading": location.heading,
                "location_updated_at": datetime.utcnow()
            }
        }
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {"status": "updated"}

@api_router.get("/device/{device_id}/status")
async def get_device_status(device_id: str):
    """Get device status including active alerts"""
    device = await db.devices.find_one({"device_id": device_id})
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Get active alert for this device
    active_alert = await db.alerts.find_one({
        "device_id": device_id,
        "status": "active"
    })
    
    # Check cooldown
    last_alert = await db.alerts.find_one(
        {"device_id": device_id},
        sort=[("timestamp", -1)]
    )
    
    can_trigger = True
    cooldown_remaining = 0
    
    if last_alert:
        time_since_last = (datetime.utcnow() - last_alert["timestamp"]).total_seconds()
        if time_since_last < COOLDOWN_SECONDS:
            can_trigger = False
            cooldown_remaining = int(COOLDOWN_SECONDS - time_since_last)
    
    return {
        "device_id": device_id,
        "is_banned": device.get("is_banned", False),
        "has_active_alert": active_alert is not None,
        "active_alert_id": active_alert["alert_id"] if active_alert else None,
        "can_trigger_alert": can_trigger,
        "cooldown_remaining": cooldown_remaining
    }

# ==================== ALERT ENDPOINTS ====================

@api_router.post("/alert/trigger", response_model=AlertResponse)
async def trigger_alert(alert_data: AlertTrigger):
    """Trigger an emergency alert"""
    # Check if device exists
    device = await db.devices.find_one({"device_id": alert_data.device_id})
    if not device:
        raise HTTPException(status_code=404, detail="Device not registered")
    
    # Check if banned
    if device.get("is_banned"):
        raise HTTPException(status_code=403, detail="Device is banned")
    
    # Check for existing active alert
    existing_alert = await db.alerts.find_one({
        "device_id": alert_data.device_id,
        "status": "active"
    })
    
    if existing_alert:
        raise HTTPException(status_code=400, detail="Already have an active alert")
    
    # Check cooldown
    last_alert = await db.alerts.find_one(
        {"device_id": alert_data.device_id},
        sort=[("timestamp", -1)]
    )
    
    if last_alert:
        time_since_last = (datetime.utcnow() - last_alert["timestamp"]).total_seconds()
        if time_since_last < COOLDOWN_SECONDS:
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {int(COOLDOWN_SECONDS - time_since_last)} seconds before triggering another alert"
            )
    
    # Create alert
    alert_id = str(uuid.uuid4())
    alert = {
        "alert_id": alert_id,
        "device_id": alert_data.device_id,
        "latitude": alert_data.latitude,
        "longitude": alert_data.longitude,
        "heading": alert_data.heading,
        "trigger_type": alert_data.trigger_type,
        "timestamp": datetime.utcnow(),
        "status": "active",
        "current_radius": RADIUS_STAGES[0],
        "location_history": [{
            "latitude": alert_data.latitude,
            "longitude": alert_data.longitude,
            "heading": alert_data.heading,
            "timestamp": datetime.utcnow()
        }],
        "notified_count": 0,
        "responder_count": 0
    }
    
    await db.alerts.insert_one(alert)
    
    # Update device alert count
    await db.devices.update_one(
        {"device_id": alert_data.device_id},
        {
            "$inc": {"alert_count": 1},
            "$set": {
                "last_latitude": alert_data.latitude,
                "last_longitude": alert_data.longitude
            }
        }
    )
    
    # Broadcast to nearby devices
    notified = await broadcast_alert_to_nearby(alert, RADIUS_STAGES[0])
    logger.info(f"Alert {alert_id} triggered, notified {notified} devices in {RADIUS_STAGES[0]}m radius")
    
    # Emit via Socket.IO
    await sio.emit('new_alert', {
        "alert_id": alert_id,
        "latitude": alert_data.latitude,
        "longitude": alert_data.longitude,
        "heading": alert_data.heading,
        "timestamp": alert["timestamp"].isoformat()
    })
    
    # Start background tasks for radius expansion and auto-end
    asyncio.create_task(expand_radius_task(alert_id))
    asyncio.create_task(auto_end_alert_task(alert_id))
    
    return AlertResponse(
        alert_id=alert_id,
        device_id=alert_data.device_id,
        latitude=alert_data.latitude,
        longitude=alert_data.longitude,
        heading=alert_data.heading,
        timestamp=alert["timestamp"],
        status="active",
        current_radius=RADIUS_STAGES[0]
    )

@api_router.post("/alert/end")
async def end_alert(alert_end: AlertEnd):
    """End an alert manually (victim taps 'I am safe')"""
    alert = await db.alerts.find_one({
        "alert_id": alert_end.alert_id,
        "device_id": alert_end.device_id
    })
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    if alert.get("status") != "active":
        raise HTTPException(status_code=400, detail="Alert is not active")
    
    await db.alerts.update_one(
        {"alert_id": alert_end.alert_id},
        {
            "$set": {
                "status": "ended",
                "ended_at": datetime.utcnow(),
                "end_reason": "user_safe"
            }
        }
    )
    
    # Emit via Socket.IO
    await sio.emit('alert_ended', {
        "alert_id": alert_end.alert_id,
        "reason": "user_safe"
    })
    
    return {"status": "ended", "reason": "user_safe"}

@api_router.put("/alert/{alert_id}/location")
async def update_alert_location(alert_id: str, location: LocationUpdate):
    """Update live location during alert (10-second tracking)"""
    alert = await db.alerts.find_one({
        "alert_id": alert_id,
        "device_id": location.device_id,
        "status": "active"
    })
    
    if not alert:
        raise HTTPException(status_code=404, detail="Active alert not found")
    
    location_entry = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "heading": location.heading,
        "timestamp": datetime.utcnow()
    }
    
    await db.alerts.update_one(
        {"alert_id": alert_id},
        {
            "$set": {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "heading": location.heading
            },
            "$push": {"location_history": location_entry}
        }
    )
    
    # Broadcast location update via Socket.IO
    await sio.emit('alert_location_updated', {
        "alert_id": alert_id,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "heading": location.heading,
        "timestamp": location_entry["timestamp"].isoformat()
    })
    
    return {"status": "updated"}

@api_router.get("/alerts/nearby")
async def get_nearby_alerts(
    latitude: float = Query(...),
    longitude: float = Query(...),
    device_id: str = Query(...)
):
    """Get active alerts near a location"""
    # Update device location and last seen
    await db.devices.update_one(
        {"device_id": device_id},
        {
            "$set": {
                "last_latitude": latitude,
                "last_longitude": longitude,
                "location_updated_at": datetime.utcnow(),
                "last_seen": datetime.utcnow()
            }
        }
    )
    
    # Get all active alerts
    active_alerts = await db.alerts.find({"status": "active"}).to_list(1000)
    
    nearby_alerts = []
    for alert in active_alerts:
        distance = calculate_distance(
            latitude, longitude,
            alert["latitude"], alert["longitude"]
        )
        
        # Only include if within current radius
        if distance <= alert.get("current_radius", 300):
            bearing = calculate_bearing(
                latitude, longitude,
                alert["latitude"], alert["longitude"]
            )
            
            nearby_alerts.append({
                "alert_id": alert["alert_id"],
                "latitude": alert["latitude"],
                "longitude": alert["longitude"],
                "heading": alert.get("heading"),
                "distance": round(distance, 1),
                "bearing": round(bearing, 1),
                "timestamp": alert["timestamp"].isoformat(),
                "current_radius": alert.get("current_radius", 300),
                "responder_count": alert.get("responder_count", 0)
            })
    
    return {"alerts": nearby_alerts}

@api_router.get("/users/nearby-count")
async def get_nearby_users_count(
    latitude: float = Query(...),
    longitude: float = Query(...),
    radius: int = Query(default=1000),
    device_id: str = Query(...)
):
    """Get count of active users nearby (within radius) - privacy-preserving"""
    # Get devices seen within the online timeout period
    online_threshold = datetime.utcnow() - timedelta(seconds=ONLINE_TIMEOUT_SECONDS)
    
    devices = await db.devices.find({
        "is_banned": {"$ne": True},
        "last_latitude": {"$exists": True},
        "last_longitude": {"$exists": True},
        "last_seen": {"$gte": online_threshold}
    }).to_list(10000)
    
    nearby_count = 0
    for device in devices:
        if device["device_id"] == device_id:
            continue
        
        dist = calculate_distance(
            latitude, longitude,
            device["last_latitude"], device["last_longitude"]
        )
        
        if dist <= radius:
            nearby_count += 1
    
    return {
        "nearby_users": nearby_count,
        "radius": radius,
        "timestamp": datetime.utcnow().isoformat()
    }

@api_router.get("/alert/{alert_id}")
async def get_alert_details(alert_id: str):
    """Get alert details including location history"""
    alert = await db.alerts.find_one({"alert_id": alert_id})
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    # Get responders
    responders = await db.alert_responses.find({"alert_id": alert_id}).to_list(100)
    
    return {
        "alert_id": alert["alert_id"],
        "latitude": alert["latitude"],
        "longitude": alert["longitude"],
        "heading": alert.get("heading"),
        "timestamp": alert["timestamp"].isoformat(),
        "status": alert["status"],
        "current_radius": alert.get("current_radius", 300),
        "location_history": [
            {
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "heading": loc.get("heading"),
                "timestamp": loc["timestamp"].isoformat()
            }
            for loc in alert.get("location_history", [])
        ],
        "responder_count": len(responders),
        "ended_at": alert.get("ended_at").isoformat() if alert.get("ended_at") else None,
        "end_reason": alert.get("end_reason")
    }

@api_router.post("/alert/respond")
async def respond_to_alert(response: RespondToAlert):
    """Mark that a user is responding to an alert"""
    alert = await db.alerts.find_one({
        "alert_id": response.alert_id,
        "status": "active"
    })
    
    if not alert:
        raise HTTPException(status_code=404, detail="Active alert not found")
    
    # Check if already responding
    existing = await db.alert_responses.find_one({
        "alert_id": response.alert_id,
        "device_id": response.device_id
    })
    
    if existing:
        return {"status": "already_responding"}
    
    # Calculate distance
    distance = calculate_distance(
        response.responder_latitude, response.responder_longitude,
        alert["latitude"], alert["longitude"]
    )
    
    # Save response
    await db.alert_responses.insert_one({
        "alert_id": response.alert_id,
        "device_id": response.device_id,
        "latitude": response.responder_latitude,
        "longitude": response.responder_longitude,
        "distance": distance,
        "timestamp": datetime.utcnow()
    })
    
    # Update responder count
    await db.alerts.update_one(
        {"alert_id": response.alert_id},
        {"$inc": {"responder_count": 1}}
    )
    
    # Notify the victim that someone is responding
    victim_device = await db.devices.find_one({"device_id": alert["device_id"]})
    if victim_device and victim_device.get("push_token"):
        await send_push_notification(
            victim_device["push_token"],
            "Help is Coming!",
            f"Someone {round(distance)}m away is responding to your alert.",
            {"type": "responder_coming", "alert_id": response.alert_id}
        )
    
    # Emit via Socket.IO
    await sio.emit('responder_added', {
        "alert_id": response.alert_id,
        "responder_count": alert.get("responder_count", 0) + 1
    })
    
    return {"status": "responding", "distance": round(distance, 1)}

# ==================== ADMIN ENDPOINTS ====================

@api_router.get("/admin/dashboard")
async def admin_dashboard():
    """Get admin dashboard statistics"""
    total_devices = await db.devices.count_documents({})
    active_devices = await db.devices.count_documents({
        "last_seen": {"$gte": datetime.utcnow() - timedelta(days=7)}
    })
    banned_devices = await db.devices.count_documents({"is_banned": True})
    
    total_alerts = await db.alerts.count_documents({})
    active_alerts = await db.alerts.count_documents({"status": "active"})
    
    # Get alerts in last 24 hours
    alerts_today = await db.alerts.count_documents({
        "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=24)}
    })
    
    # Get total responders
    total_responses = await db.alert_responses.count_documents({})
    
    return {
        "devices": {
            "total": total_devices,
            "active_7_days": active_devices,
            "banned": banned_devices
        },
        "alerts": {
            "total": total_alerts,
            "active": active_alerts,
            "today": alerts_today
        },
        "responses": {
            "total": total_responses
        }
    }

@api_router.get("/admin/alerts")
async def admin_get_alerts(
    status: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    skip: int = Query(default=0)
):
    """Get alert history for admin"""
    query = {}
    if status:
        query["status"] = status
    
    alerts = await db.alerts.find(query).sort("timestamp", -1).skip(skip).limit(limit).to_list(limit)
    
    return {
        "alerts": [
            {
                "alert_id": a["alert_id"],
                "device_id": a["device_id"],
                "latitude": a["latitude"],
                "longitude": a["longitude"],
                "timestamp": a["timestamp"].isoformat(),
                "status": a["status"],
                "current_radius": a.get("current_radius", 300),
                "notified_count": a.get("notified_count", 0),
                "responder_count": a.get("responder_count", 0),
                "trigger_type": a.get("trigger_type", "button"),
                "ended_at": a.get("ended_at").isoformat() if a.get("ended_at") else None,
                "end_reason": a.get("end_reason")
            }
            for a in alerts
        ],
        "total": await db.alerts.count_documents(query)
    }

@api_router.get("/admin/devices")
async def admin_get_devices(
    limit: int = Query(default=100, le=1000),
    skip: int = Query(default=0)
):
    """Get all registered devices"""
    devices = await db.devices.find().sort("registered_at", -1).skip(skip).limit(limit).to_list(limit)
    
    return {
        "devices": [
            {
                "device_id": d["device_id"],
                "platform": d.get("platform", "unknown"),
                "registered_at": d["registered_at"].isoformat(),
                "last_seen": d.get("last_seen").isoformat() if d.get("last_seen") else None,
                "is_banned": d.get("is_banned", False),
                "alert_count": d.get("alert_count", 0),
                "has_push_token": bool(d.get("push_token"))
            }
            for d in devices
        ],
        "total": await db.devices.count_documents({})
    }

@api_router.post("/admin/ban")
async def admin_ban_device(ban: AdminBan):
    """Ban a device"""
    result = await db.devices.update_one(
        {"device_id": ban.device_id},
        {
            "$set": {
                "is_banned": True,
                "banned_at": datetime.utcnow(),
                "ban_reason": ban.reason
            }
        }
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {"status": "banned", "device_id": ban.device_id}

@api_router.post("/admin/unban")
async def admin_unban_device(device_id: str = Query(...)):
    """Unban a device"""
    result = await db.devices.update_one(
        {"device_id": device_id},
        {
            "$set": {"is_banned": False},
            "$unset": {"banned_at": "", "ban_reason": ""}
        }
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Device not found")
    
    return {"status": "unbanned", "device_id": device_id}

@api_router.get("/admin/heatmap")
async def admin_get_heatmap():
    """Get alert locations for heatmap"""
    alerts = await db.alerts.find({}).to_list(10000)
    
    return {
        "points": [
            {
                "latitude": a["latitude"],
                "longitude": a["longitude"],
                "timestamp": a["timestamp"].isoformat()
            }
            for a in alerts
        ]
    }

# ==================== SOCKET.IO EVENTS ====================

@sio.event
async def connect(sid, environ):
    logger.info(f"Socket connected: {sid}")

@sio.event
async def disconnect(sid):
    logger.info(f"Socket disconnected: {sid}")

@sio.event
async def join_device_room(sid, data):
    """Join a room specific to the device for targeted notifications"""
    device_id = data.get("device_id")
    if device_id:
        await sio.enter_room(sid, device_id)
        logger.info(f"Device {device_id} joined room")

@sio.event
async def leave_device_room(sid, data):
    """Leave device room"""
    device_id = data.get("device_id")
    if device_id:
        await sio.leave_room(sid, device_id)
        logger.info(f"Device {device_id} left room")

@sio.event
async def subscribe_alert(sid, data):
    """Subscribe to updates for a specific alert"""
    alert_id = data.get("alert_id")
    if alert_id:
        await sio.enter_room(sid, f"alert_{alert_id}")
        logger.info(f"Subscribed to alert {alert_id}")

# ==================== ADMIN DASHBOARD HTML ====================

ADMIN_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Alert - Admin Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body { background-color: #0f0f0f; }
        #map { height: 400px; width: 100%; border-radius: 12px; }
        .stat-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
        .alert-badge { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body class="text-white min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-red-500 flex items-center gap-3">
                <span class="text-4xl">🚨</span> Human Alert Admin
            </h1>
            <p class="text-gray-400 mt-2">Emergency Alert System Dashboard</p>
        </header>

        <!-- Stats Grid -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="stat-card rounded-xl p-6">
                <div class="text-gray-400 text-sm">Total Devices</div>
                <div id="total-devices" class="text-3xl font-bold text-white mt-2">-</div>
            </div>
            <div class="stat-card rounded-xl p-6">
                <div class="text-gray-400 text-sm">Active Alerts</div>
                <div id="active-alerts" class="text-3xl font-bold text-red-500 mt-2">-</div>
            </div>
            <div class="stat-card rounded-xl p-6">
                <div class="text-gray-400 text-sm">Alerts Today</div>
                <div id="alerts-today" class="text-3xl font-bold text-yellow-500 mt-2">-</div>
            </div>
            <div class="stat-card rounded-xl p-6">
                <div class="text-gray-400 text-sm">Banned Devices</div>
                <div id="banned-devices" class="text-3xl font-bold text-gray-500 mt-2">-</div>
            </div>
        </div>

        <!-- Map and Alerts -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Map -->
            <div class="stat-card rounded-xl p-6">
                <h2 class="text-xl font-semibold mb-4">Alert Heatmap</h2>
                <div id="map"></div>
            </div>

            <!-- Recent Alerts -->
            <div class="stat-card rounded-xl p-6">
                <h2 class="text-xl font-semibold mb-4">Recent Alerts</h2>
                <div id="alerts-list" class="space-y-3 max-h-96 overflow-y-auto">
                    <p class="text-gray-400">Loading...</p>
                </div>
            </div>
        </div>

        <!-- Devices Table -->
        <div class="stat-card rounded-xl p-6">
            <h2 class="text-xl font-semibold mb-4">Registered Devices</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-left">
                    <thead class="border-b border-gray-700">
                        <tr>
                            <th class="pb-3 text-gray-400">Device ID</th>
                            <th class="pb-3 text-gray-400">Platform</th>
                            <th class="pb-3 text-gray-400">Registered</th>
                            <th class="pb-3 text-gray-400">Alerts</th>
                            <th class="pb-3 text-gray-400">Status</th>
                            <th class="pb-3 text-gray-400">Actions</th>
                        </tr>
                    </thead>
                    <tbody id="devices-table">
                        <tr><td colspan="6" class="py-4 text-gray-400">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = '/api';
        let map;
        let markers = [];

        // Initialize map
        function initMap() {
            map = L.map('map').setView([9.0579, 7.4951], 10); // Abuja coordinates
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(map);
        }

        // Fetch dashboard stats
        async function fetchStats() {
            try {
                const res = await fetch(`${API_BASE}/admin/dashboard`);
                const data = await res.json();
                
                document.getElementById('total-devices').textContent = data.devices.total;
                document.getElementById('active-alerts').textContent = data.alerts.active;
                document.getElementById('alerts-today').textContent = data.alerts.today;
                document.getElementById('banned-devices').textContent = data.devices.banned;
            } catch (err) {
                console.error('Failed to fetch stats:', err);
            }
        }

        // Fetch alerts
        async function fetchAlerts() {
            try {
                const res = await fetch(`${API_BASE}/admin/alerts?limit=20`);
                const data = await res.json();
                
                const container = document.getElementById('alerts-list');
                if (data.alerts.length === 0) {
                    container.innerHTML = '<p class="text-gray-400">No alerts yet</p>';
                    return;
                }

                container.innerHTML = data.alerts.map(alert => `
                    <div class="bg-gray-800/50 rounded-lg p-4 ${alert.status === 'active' ? 'border border-red-500' : ''}">
                        <div class="flex justify-between items-start">
                            <div>
                                <span class="text-xs px-2 py-1 rounded ${alert.status === 'active' ? 'bg-red-500 alert-badge' : 'bg-gray-600'}">
                                    ${alert.status.toUpperCase()}
                                </span>
                                <p class="text-sm text-gray-400 mt-2">${new Date(alert.timestamp).toLocaleString()}</p>
                            </div>
                            <div class="text-right text-sm">
                                <p class="text-gray-300">${alert.notified_count} notified</p>
                                <p class="text-green-400">${alert.responder_count} responding</p>
                            </div>
                        </div>
                        <p class="text-xs text-gray-500 mt-2">ID: ${alert.alert_id.slice(0, 8)}...</p>
                    </div>
                `).join('');
            } catch (err) {
                console.error('Failed to fetch alerts:', err);
            }
        }

        // Fetch devices
        async function fetchDevices() {
            try {
                const res = await fetch(`${API_BASE}/admin/devices?limit=50`);
                const data = await res.json();
                
                const tbody = document.getElementById('devices-table');
                if (data.devices.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="py-4 text-gray-400">No devices registered</td></tr>';
                    return;
                }

                tbody.innerHTML = data.devices.map(device => `
                    <tr class="border-b border-gray-800">
                        <td class="py-3 text-sm font-mono">${device.device_id.slice(0, 12)}...</td>
                        <td class="py-3">${device.platform}</td>
                        <td class="py-3 text-sm text-gray-400">${new Date(device.registered_at).toLocaleDateString()}</td>
                        <td class="py-3">${device.alert_count}</td>
                        <td class="py-3">
                            <span class="text-xs px-2 py-1 rounded ${device.is_banned ? 'bg-red-900 text-red-300' : 'bg-green-900 text-green-300'}">
                                ${device.is_banned ? 'BANNED' : 'ACTIVE'}
                            </span>
                        </td>
                        <td class="py-3">
                            ${device.is_banned ? 
                                `<button onclick="unbanDevice('${device.device_id}')" class="text-sm text-green-400 hover:text-green-300">Unban</button>` :
                                `<button onclick="banDevice('${device.device_id}')" class="text-sm text-red-400 hover:text-red-300">Ban</button>`
                            }
                        </td>
                    </tr>
                `).join('');
            } catch (err) {
                console.error('Failed to fetch devices:', err);
            }
        }

        // Fetch heatmap data
        async function fetchHeatmap() {
            try {
                const res = await fetch(`${API_BASE}/admin/heatmap`);
                const data = await res.json();
                
                // Clear existing markers
                markers.forEach(m => map.removeLayer(m));
                markers = [];
                
                // Add markers for each alert
                data.points.forEach(point => {
                    const marker = L.circleMarker([point.latitude, point.longitude], {
                        radius: 8,
                        fillColor: '#ef4444',
                        color: '#dc2626',
                        weight: 2,
                        opacity: 0.8,
                        fillOpacity: 0.6
                    }).addTo(map);
                    markers.push(marker);
                });
                
                // Fit bounds if we have points
                if (data.points.length > 0) {
                    const bounds = L.latLngBounds(data.points.map(p => [p.latitude, p.longitude]));
                    map.fitBounds(bounds, { padding: [50, 50] });
                }
            } catch (err) {
                console.error('Failed to fetch heatmap:', err);
            }
        }

        // Ban device
        async function banDevice(deviceId) {
            const reason = prompt('Enter ban reason:');
            if (!reason) return;
            
            try {
                await fetch(`${API_BASE}/admin/ban`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ device_id: deviceId, reason })
                });
                fetchDevices();
                fetchStats();
            } catch (err) {
                alert('Failed to ban device');
            }
        }

        // Unban device
        async function unbanDevice(deviceId) {
            try {
                await fetch(`${API_BASE}/admin/unban?device_id=${deviceId}`, { method: 'POST' });
                fetchDevices();
                fetchStats();
            } catch (err) {
                alert('Failed to unban device');
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initMap();
            fetchStats();
            fetchAlerts();
            fetchDevices();
            fetchHeatmap();
            
            // Refresh data every 10 seconds
            setInterval(() => {
                fetchStats();
                fetchAlerts();
                fetchHeatmap();
            }, 10000);
        });
    </script>
</body>
</html>
"""

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve admin dashboard HTML"""
    return ADMIN_DASHBOARD_HTML

@api_router.get("/admin-dashboard", response_class=HTMLResponse)
async def admin_dashboard_api():
    """Serve admin dashboard HTML via API route"""
    return ADMIN_DASHBOARD_HTML

@api_router.get("/download/human-alert-app.zip")
async def download_zip():
    """Download the Human Alert source code ZIP"""
    zip_path = "/tmp/human-alert-app.zip"
    if os.path.exists(zip_path):
        return FileResponse(
            path=zip_path,
            filename="human-alert-app.zip",
            media_type="application/zip"
        )
    raise HTTPException(status_code=404, detail="ZIP file not found")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# For running with Socket.IO
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8001)
