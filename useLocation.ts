import { useState, useEffect, useCallback, useRef } from 'react';
import * as Location from 'expo-location';
import { Platform } from 'react-native';
import { useAlertStore } from '../store/alertStore';

export const useLocation = () => {
  const [hasPermission, setHasPermission] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const setCurrentLocation = useAlertStore((state) => state.setCurrentLocation);
  const watchSubscription = useRef<Location.LocationSubscription | null>(null);

  const requestPermission = useCallback(async (): Promise<boolean> => {
    try {
      // Skip on web
      if (Platform.OS === 'web') {
        // Try to use browser geolocation
        if (navigator.geolocation) {
          return new Promise((resolve) => {
            navigator.geolocation.getCurrentPosition(
              () => {
                setHasPermission(true);
                resolve(true);
              },
              () => {
                setHasPermission(false);
                resolve(false);
              }
            );
          });
        }
        setHasPermission(false);
        return false;
      }

      const { status } = await Location.requestForegroundPermissionsAsync();
      const granted = status === 'granted';
      setHasPermission(granted);
      
      if (!granted) {
        setError('Location permission denied');
      }
      
      return granted;
    } catch (err) {
      console.error('Error requesting location permission:', err);
      setError('Failed to request location permission');
      return false;
    }
  }, []);

  const getCurrentLocation = useCallback(async () => {
    try {
      // Web browser geolocation
      if (Platform.OS === 'web') {
        return new Promise<{ latitude: number; longitude: number; heading?: number } | null>((resolve) => {
          if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
              (position) => {
                const locationData = {
                  latitude: position.coords.latitude,
                  longitude: position.coords.longitude,
                  heading: position.coords.heading || undefined,
                };
                setCurrentLocation(locationData);
                resolve(locationData);
              },
              (error) => {
                console.error('Web geolocation error:', error);
                resolve(null);
              },
              { enableHighAccuracy: true, timeout: 10000 }
            );
          } else {
            resolve(null);
          }
        });
      }

      // Native location
      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
      });

      let heading: number | undefined;
      try {
        const headingData = await Location.getHeadingAsync();
        heading = headingData?.trueHeading || headingData?.magHeading;
      } catch (e) {
        // Heading not available
      }

      const locationData = {
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
        heading,
      };

      setCurrentLocation(locationData);
      return locationData;
    } catch (err) {
      console.error('Error getting current location:', err);
      setError('Failed to get current location');
      return null;
    }
  }, [setCurrentLocation]);

  const startWatching = useCallback(async (onUpdate: (location: any) => void) => {
    if (Platform.OS === 'web') return null;

    try {
      const subscription = await Location.watchPositionAsync(
        {
          accuracy: Location.Accuracy.High,
          timeInterval: 2000,
          distanceInterval: 5,
        },
        async (location) => {
          let heading: number | undefined;
          try {
            const headingData = await Location.getHeadingAsync();
            heading = headingData?.trueHeading || headingData?.magHeading;
          } catch (e) {}

          const locationData = {
            latitude: location.coords.latitude,
            longitude: location.coords.longitude,
            heading,
          };
          
          setCurrentLocation(locationData);
          onUpdate(locationData);
        }
      );
      
      watchSubscription.current = subscription;
      return subscription;
    } catch (err) {
      console.error('Error starting location watch:', err);
      return null;
    }
  }, [setCurrentLocation]);

  const stopWatching = useCallback(() => {
    if (watchSubscription.current) {
      watchSubscription.current.remove();
      watchSubscription.current = null;
    }
  }, []);

  // Request permission on mount
  useEffect(() => {
    requestPermission();
    
    return () => {
      stopWatching();
    };
  }, [requestPermission, stopWatching]);

  return {
    hasPermission,
    error,
    requestPermission,
    getCurrentLocation,
    startWatching,
    stopWatching,
  };
};
