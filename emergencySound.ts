import { Audio } from 'expo-av';
import { Platform, Vibration } from 'react-native';

// Track if sound is initialized
let soundObject: Audio.Sound | null = null;
let isInitialized = false;
let isPlaying = false;

// Emergency vibration pattern
const EMERGENCY_VIBRATION_PATTERN = [0, 400, 150, 400, 150, 400, 150, 400];

// Initialize audio settings once (call on app start)
export async function initializeAudio(): Promise<void> {
  if (isInitialized || Platform.OS === 'web') return;
  
  try {
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: false,
      staysActiveInBackground: true,
      playsInSilentModeIOS: true,
      shouldDuckAndroid: false,
      playThroughEarpieceAndroid: false,
    });
    isInitialized = true;
    console.log('Audio mode initialized for emergency alerts');
  } catch (error) {
    console.error('Failed to initialize audio:', error);
  }
}

// Play emergency alert - NON-BLOCKING with proper sound
export function playAlertBurst(): void {
  // Fire vibration immediately - synchronous, never blocks
  if (Platform.OS !== 'web') {
    try {
      Vibration.vibrate(EMERGENCY_VIBRATION_PATTERN);
    } catch (e) {
      console.error('Vibration error:', e);
    }
  }
  
  // Play sound asynchronously - don't await, don't block
  if (Platform.OS !== 'web') {
    playEmergencySoundAsync();
  }
}

// Async sound playback - runs in background
async function playEmergencySoundAsync(): Promise<void> {
  if (isPlaying) return;
  
  try {
    isPlaying = true;
    
    // Ensure audio is initialized
    if (!isInitialized) {
      await initializeAudio();
    }
    
    // Unload previous sound if exists
    if (soundObject) {
      try {
        await soundObject.unloadAsync();
      } catch (e) {
        // Ignore unload errors
      }
      soundObject = null;
    }
    
    // Create and play new sound - using a reliable emergency sound URL
    const { sound } = await Audio.Sound.createAsync(
      { uri: 'https://cdn.pixabay.com/audio/2022/03/15/audio_115b9d1791.mp3' }, // Emergency alarm sound
      { 
        shouldPlay: true,
        volume: 1.0,
        isLooping: false,
      }
    );
    
    soundObject = sound;
    
    // Auto-cleanup after 4 seconds
    setTimeout(async () => {
      try {
        if (soundObject) {
          await soundObject.stopAsync();
          await soundObject.unloadAsync();
          soundObject = null;
        }
      } catch (e) {
        // Ignore cleanup errors
      }
      isPlaying = false;
    }, 4000);
    
    console.log('Emergency sound playing');
  } catch (error) {
    console.error('Error playing emergency sound:', error);
    isPlaying = false;
  }
}

// Play continuous emergency alert (for responder view)
export function playEmergencyAlert(): void {
  playAlertBurst();
}

// Stop emergency alert
export function stopEmergencyAlert(): void {
  try {
    isPlaying = false;
    
    if (Platform.OS !== 'web') {
      Vibration.cancel();
    }
    
    if (soundObject) {
      soundObject.stopAsync().catch(() => {});
      soundObject.unloadAsync().catch(() => {});
      soundObject = null;
    }
    
    console.log('Emergency alert stopped');
  } catch (error) {
    console.error('Error stopping emergency alert:', error);
  }
}

// Cleanup
export function unloadEmergencySound(): void {
  stopEmergencyAlert();
  isInitialized = false;
}

export { isPlaying };
