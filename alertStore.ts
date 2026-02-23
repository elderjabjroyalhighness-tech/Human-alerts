import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface Alert {
  alert_id: string;
  latitude: number;
  longitude: number;
  heading?: number;
  distance: number;
  bearing: number;
  timestamp: string;
  current_radius: number;
  responder_count: number;
}

export interface MyAlert {
  alert_id: string;
  latitude: number;
  longitude: number;
}

export interface Location {
  latitude: number;
  longitude: number;
  heading?: number;
}

interface AlertStore {
  // Device info
  deviceId: string | null;
  pushToken: string | null;
  isRegistered: boolean;
  
  // Alert states
  myActiveAlert: MyAlert | null; // Alert I triggered
  nearbyAlerts: Alert[]; // Alerts from others
  selectedAlert: Alert | null; // Currently viewing
  
  // Trigger states
  canTrigger: boolean;
  cooldownRemaining: number;
  
  // Location
  currentLocation: Location | null;
  
  // UI states
  isLoading: boolean;
  error: string | null;
  isConnected: boolean;
  
  // Actions
  setDeviceId: (id: string) => void;
  setPushToken: (token: string | null) => void;
  setIsRegistered: (registered: boolean) => void;
  setMyActiveAlert: (alert: MyAlert | null) => void;
  setNearbyAlerts: (alerts: Alert[]) => void;
  setSelectedAlert: (alert: Alert | null) => void;
  setCanTrigger: (can: boolean) => void;
  setCooldownRemaining: (seconds: number) => void;
  setCurrentLocation: (location: Location | null) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setIsConnected: (connected: boolean) => void;
  loadDeviceId: () => Promise<string>;
  reset: () => void;
}

export const useAlertStore = create<AlertStore>((set, get) => ({
  deviceId: null,
  pushToken: null,
  isRegistered: false,
  myActiveAlert: null,
  nearbyAlerts: [],
  selectedAlert: null,
  canTrigger: true,
  cooldownRemaining: 0,
  currentLocation: null,
  isLoading: false,
  error: null,
  isConnected: false,

  setDeviceId: (id) => set({ deviceId: id }),
  setPushToken: (token) => set({ pushToken: token }),
  setIsRegistered: (registered) => set({ isRegistered: registered }),
  setMyActiveAlert: (alert) => set({ myActiveAlert: alert }),
  setNearbyAlerts: (alerts) => set({ nearbyAlerts: alerts }),
  setSelectedAlert: (alert) => set({ selectedAlert: alert }),
  setCanTrigger: (can) => set({ canTrigger: can }),
  setCooldownRemaining: (seconds) => set({ cooldownRemaining: seconds }),
  setCurrentLocation: (location) => set({ currentLocation: location }),
  setIsLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error: error }),
  setIsConnected: (connected) => set({ isConnected: connected }),
  
  loadDeviceId: async () => {
    try {
      let storedId = await AsyncStorage.getItem('human_alert_device_id');
      if (!storedId) {
        // Generate unique device ID
        storedId = 'ha_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now().toString(36);
        await AsyncStorage.setItem('human_alert_device_id', storedId);
      }
      set({ deviceId: storedId });
      return storedId;
    } catch (error) {
      console.error('Error loading device ID:', error);
      const fallbackId = 'ha_' + Math.random().toString(36).substring(2, 10);
      set({ deviceId: fallbackId });
      return fallbackId;
    }
  },
  
  reset: () => set({
    myActiveAlert: null,
    nearbyAlerts: [],
    selectedAlert: null,
    canTrigger: true,
    cooldownRemaining: 0,
    error: null,
  }),
}));
