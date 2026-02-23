import axios from 'axios';

// Backend URL - uses environment variable or fallback
const API_BASE = process.env.EXPO_PUBLIC_BACKEND_URL || 'https://safe-radius.preview.emergentagent.com';

console.log('API Base URL:', API_BASE);

const api = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Types
export interface DeviceRegisterRequest {
  device_id: string;
  platform: string;
  push_token?: string;
}

export interface LocationUpdate {
  device_id: string;
  latitude: number;
  longitude: number;
  heading?: number;
}

export interface AlertTrigger {
  device_id: string;
  latitude: number;
  longitude: number;
  heading?: number;
  trigger_type: 'button' | 'power_button';
}

export interface AlertEnd {
  device_id: string;
  alert_id: string;
}

export interface RespondToAlert {
  device_id: string;
  alert_id: string;
  responder_latitude: number;
  responder_longitude: number;
}

// Device APIs
export const registerDevice = async (data: DeviceRegisterRequest) => {
  const response = await api.post('/device/register', data);
  return response.data;
};

export const updateDeviceLocation = async (data: LocationUpdate) => {
  const response = await api.put('/device/location', data);
  return response.data;
};

export const getDeviceStatus = async (deviceId: string) => {
  const response = await api.get(`/device/${deviceId}/status`);
  return response.data;
};

// Alert APIs
export const triggerAlert = async (data: AlertTrigger) => {
  const response = await api.post('/alert/trigger', data);
  return response.data;
};

export const endAlert = async (data: AlertEnd) => {
  const response = await api.post('/alert/end', data);
  return response.data;
};

export const updateAlertLocation = async (alertId: string, data: LocationUpdate) => {
  const response = await api.put(`/alert/${alertId}/location`, data);
  return response.data;
};

export const getNearbyAlerts = async (latitude: number, longitude: number, deviceId: string) => {
  const response = await api.get('/alerts/nearby', {
    params: { latitude, longitude, device_id: deviceId },
  });
  return response.data;
};

export const getAlertDetails = async (alertId: string) => {
  const response = await api.get(`/alert/${alertId}`);
  return response.data;
};

export const respondToAlert = async (data: RespondToAlert) => {
  const response = await api.post('/alert/respond', data);
  return response.data;
};

// Get count of nearby online users (privacy-preserving)
export const getNearbyUsersCount = async (
  latitude: number,
  longitude: number,
  deviceId: string,
  radius: number = 1000
) => {
  const response = await api.get('/users/nearby-count', {
    params: { latitude, longitude, device_id: deviceId, radius },
  });
  return response.data;
};

export { API_BASE };
export default api;
