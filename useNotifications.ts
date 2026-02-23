import { useState, useEffect, useRef, useCallback } from 'react';
import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import { Platform, Vibration } from 'react-native';
import { useAlertStore } from '../store/alertStore';
import { router } from 'expo-router';
import { playAlertBurst, initializeAudio } from '../services/emergencySound';

// Configure notification handler - ensure sound plays
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
    priority: Notifications.AndroidNotificationPriority.MAX,
  }),
});

export const useNotifications = () => {
  const [expoPushToken, setExpoPushToken] = useState<string | null>(null);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const notificationListener = useRef<Notifications.Subscription | null>(null);
  const responseListener = useRef<Notifications.Subscription | null>(null);
  const setPushToken = useAlertStore((state) => state.setPushToken);
  const setSelectedAlert = useAlertStore((state) => state.setSelectedAlert);

  const registerForPushNotifications = useCallback(async (): Promise<string | null> => {
    // Skip on web
    if (Platform.OS === 'web') {
      console.log('Push notifications not supported on web');
      return null;
    }

    try {
      // Create Android notification channel with MAXIMUM importance and LOUD sound
      if (Platform.OS === 'android') {
        // Delete old channel first to apply new settings
        await Notifications.deleteNotificationChannelAsync('emergency-alerts').catch(() => {});
        
        await Notifications.setNotificationChannelAsync('emergency-alerts', {
          name: 'Emergency Alerts',
          description: 'Critical emergency alerts that require immediate attention',
          importance: Notifications.AndroidImportance.MAX,
          vibrationPattern: [0, 500, 200, 500, 200, 500, 200, 500],
          lightColor: '#FF0000',
          sound: 'default',  // Uses system default alarm sound
          enableVibrate: true,
          enableLights: true,
          lockscreenVisibility: Notifications.AndroidNotificationVisibility.PUBLIC,
          bypassDnd: true,  // Bypass Do Not Disturb
          showBadge: true,
        });
        
        console.log('Android notification channel created with MAX importance');
      }

      // Check for physical device
      if (!Device.isDevice) {
        console.log('Push notifications require physical device');
        return null;
      }

      // Request permissions
      const { status: existingStatus } = await Notifications.getPermissionsAsync();
      let finalStatus = existingStatus;

      if (existingStatus !== 'granted') {
        const { status } = await Notifications.requestPermissionsAsync();
        finalStatus = status;
      }

      if (finalStatus !== 'granted') {
        console.log('Push notification permission not granted');
        setPermissionGranted(false);
        return null;
      }

      setPermissionGranted(true);

      // Get push token
      const tokenData = await Notifications.getExpoPushTokenAsync({
        projectId: 'human-alert-emergency',
      });

      const token = tokenData.data;
      console.log('Expo Push Token:', token);

      setExpoPushToken(token);
      setPushToken(token);

      return token;
    } catch (error) {
      console.error('Error registering for push notifications:', error);
      return null;
    }
  }, [setPushToken]);

  // Send local notification (for testing and immediate alerts) with LOUD sound
  const sendLocalNotification = useCallback(async (
    title: string,
    body: string,
    data?: Record<string, any>
  ): Promise<void> => {
    if (Platform.OS === 'web') return;

    try {
      // Vibrate immediately with emergency pattern
      Vibration.vibrate([0, 500, 200, 500, 200, 500, 200, 500]);

      // Also play sound via expo-av
      playAlertBurst();

      await Notifications.scheduleNotificationAsync({
        content: {
          title,
          body,
          data: data || {},
          sound: 'default',  // Ensures sound plays
          priority: Notifications.AndroidNotificationPriority.MAX,
          vibrate: [0, 500, 200, 500, 200, 500],
          sticky: true,  // Notification stays until dismissed
        },
        trigger: null, // Immediate
      });
    } catch (error) {
      console.error('Error sending local notification:', error);
    }
  }, []);

  useEffect(() => {
    if (Platform.OS === 'web') return;

    registerForPushNotifications();

    // Foreground notification listener
    notificationListener.current = Notifications.addNotificationReceivedListener((notification) => {
      const data = notification.request.content.data;
      console.log('Notification received:', data);

      // Play emergency sound and vibrate
      playAlertBurst();

      if (data?.type === 'emergency_alert') {
        setSelectedAlert({
          alert_id: data.alert_id as string,
          latitude: data.latitude as number,
          longitude: data.longitude as number,
          distance: (data.distance as number) || 0,
          bearing: 0,
          timestamp: new Date().toISOString(),
          current_radius: 300,
          responder_count: 0,
        });
      }
    });

    // Notification tap listener
    responseListener.current = Notifications.addNotificationResponseReceivedListener((response) => {
      const data = response.notification.request.content.data;
      console.log('Notification tapped:', data);

      if (data?.type === 'emergency_alert' && data.alert_id) {
        router.push({
          pathname: '/alert-view',
          params: { alertId: data.alert_id as string },
        });
      }
    });

    return () => {
      if (notificationListener.current) {
        Notifications.removeNotificationSubscription(notificationListener.current);
      }
      if (responseListener.current) {
        Notifications.removeNotificationSubscription(responseListener.current);
      }
    };
  }, [registerForPushNotifications, setSelectedAlert]);

  return {
    expoPushToken,
    permissionGranted,
    registerForPushNotifications,
    sendLocalNotification,
  };
};
