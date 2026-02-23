import { io, Socket } from 'socket.io-client';
import { playAlertBurst } from './emergencySound';

const API_BASE = process.env.EXPO_PUBLIC_BACKEND_URL || 'https://safe-radius.preview.emergentagent.com';

class SocketService {
  private socket: Socket | null = null;
  private deviceId: string | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;

  connect(deviceId: string): void {
    if (this.socket?.connected && this.deviceId === deviceId) {
      console.log('Socket already connected for device:', deviceId);
      return;
    }

    // Disconnect existing socket if any
    if (this.socket) {
      this.socket.disconnect();
    }

    this.deviceId = deviceId;
    console.log('Connecting socket for device:', deviceId);

    this.socket = io(API_BASE, {
      transports: ['websocket'],  // WebSocket only for fastest delivery
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: 500,      // Reduced from 1000ms
      reconnectionDelayMax: 2000,  // Reduced from 5000ms
      timeout: 10000,              // Reduced from 20000ms
      forceNew: true,
    });

    this.socket.on('connect', () => {
      console.log('Socket connected:', this.socket?.id);
      this.reconnectAttempts = 0;
      this.joinDeviceRoom();
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Socket disconnected:', reason);
      // Immediately attempt reconnect for critical disconnects
      if (reason === 'io server disconnect' || reason === 'transport close') {
        this.socket?.connect();
      }
    });

    this.socket.on('connect_error', (error) => {
      console.log('Socket connection error:', error.message);
      this.reconnectAttempts++;
    });

    this.socket.on('reconnect', (attemptNumber) => {
      console.log('Socket reconnected after', attemptNumber, 'attempts');
      this.joinDeviceRoom();
    });
  }

  disconnect(): void {
    if (this.socket) {
      console.log('Disconnecting socket');
      this.socket.disconnect();
      this.socket = null;
    }
  }

  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  private joinDeviceRoom(): void {
    if (this.socket && this.deviceId) {
      this.socket.emit('join_device_room', { device_id: this.deviceId });
      console.log('Joined device room:', this.deviceId);
    }
  }

  subscribeToAlert(alertId: string): void {
    if (this.socket) {
      this.socket.emit('subscribe_alert', { alert_id: alertId });
      console.log('Subscribed to alert:', alertId);
    }
  }

  // Event listeners - instant delivery with sound
  onAlertReceived(callback: (data: any) => void): void {
    this.socket?.on('alert_received', (data) => {
      console.log('ALERT RECEIVED via socket:', data.alert_id);
      // Play emergency sound IMMEDIATELY
      playAlertBurst();
      callback(data);
    });
  }

  onNewAlert(callback: (data: any) => void): void {
    this.socket?.on('new_alert', (data) => {
      console.log('NEW ALERT via socket:', data.alert_id);
      // Play emergency sound when new alert is received
      playAlertBurst();
      callback(data);
    });
  }

  onAlertLocationUpdated(callback: (data: any) => void): void {
    this.socket?.on('alert_location_updated', callback);
  }

  onAlertEnded(callback: (data: any) => void): void {
    this.socket?.on('alert_ended', callback);
  }

  onRadiusExpanded(callback: (data: any) => void): void {
    this.socket?.on('radius_expanded', callback);
  }

  onResponderAdded(callback: (data: any) => void): void {
    this.socket?.on('responder_added', callback);
  }

  removeListener(event: string): void {
    this.socket?.off(event);
  }

  removeAllListeners(): void {
    if (this.socket) {
      this.socket.removeAllListeners();
      // Re-add connection listeners
      this.socket.on('connect', () => this.joinDeviceRoom());
    }
  }
}

export const socketService = new SocketService();
