# Human Alert - Android APK Build Guide

## 📱 Build a Standalone APK for Direct Installation

This guide will help you build a standalone Android APK file that can be installed directly on any Android device without the Play Store.

---

## ✅ Prerequisites

Before building, ensure you have:

1. **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
2. **Git** - [Download](https://git-scm.com/)
3. **Expo Account** - [Create Free Account](https://expo.dev/signup)

---

## 🚀 Quick Start (5 Steps)

### Step 1: Install EAS CLI

Open your terminal and run:

```bash
npm install -g eas-cli
```

### Step 2: Login to Expo

```bash
eas login
```

Enter your Expo account email and password when prompted.

### Step 3: Download the Project

Download this project or clone it to your local machine.

### Step 4: Navigate to Frontend Folder

```bash
cd frontend
```

### Step 5: Build the APK

Run this command to build the APK:

```bash
eas build --platform android --profile preview
```

**What happens:**
- EAS uploads your code to Expo's build servers
- Build takes approximately 10-15 minutes
- You'll receive a download link for the APK

---

## 📥 Download Your APK

After the build completes:

1. Check your terminal for the download URL
2. Or visit [Expo Dashboard](https://expo.dev) → Your Project → Builds
3. Click **Download** to get the `.apk` file

---

## 📲 Install on Android Device

### Option A: Direct Install
1. Transfer the APK to your Android device
2. Open the APK file
3. Allow "Install from Unknown Sources" if prompted
4. Tap **Install**

### Option B: Via ADB (Developer Mode)
```bash
adb install human-alert.apk
```

---

## 🔧 Build Profiles Explained

| Profile | Command | Output | Use Case |
|---------|---------|--------|----------|
| `preview` | `eas build --profile preview` | `.apk` | Testing & Internal distribution |
| `development` | `eas build --profile development` | `.apk` (debug) | Development with Expo Dev Client |
| `production` | `eas build --profile production` | `.apk` | Final release build |

**Recommended:** Use `preview` profile for testing.

---

## 📋 Permissions Included

The APK includes these Android permissions:

| Permission | Purpose |
|------------|---------|
| `ACCESS_FINE_LOCATION` | GPS for precise location |
| `ACCESS_COARSE_LOCATION` | Network-based location |
| `ACCESS_BACKGROUND_LOCATION` | Location while app is in background |
| `FOREGROUND_SERVICE` | Keep app running for alerts |
| `POST_NOTIFICATIONS` | Send push notifications |
| `VIBRATE` | Emergency vibration alerts |
| `WAKE_LOCK` | Keep device awake for alerts |
| `INTERNET` | Network connectivity |

---

## ⚙️ Configuration Files

### eas.json (Already Configured)
```json
{
  "build": {
    "preview": {
      "distribution": "internal",
      "android": {
        "buildType": "apk"  // Outputs .apk NOT .aab
      }
    }
  }
}
```

### app.json Highlights
- **Package Name:** `com.humanalert.app`
- **Version:** 1.0.0
- **Min SDK:** Android 5.0+ (API 21)

---

## 🔄 Updating the Backend URL

The APK connects to:
```
https://safe-radius.preview.emergentagent.com
```

To change this, edit `app.json`:
```json
{
  "extra": {
    "backendUrl": "YOUR_NEW_BACKEND_URL"
  }
}
```

Then rebuild the APK.

---

## 🐛 Troubleshooting

### "eas: command not found"
```bash
npm install -g eas-cli
```

### "Not logged in"
```bash
eas login
```

### Build fails with dependency errors
```bash
cd frontend
rm -rf node_modules
npm install
eas build --platform android --profile preview
```

### APK won't install
- Enable "Unknown Sources" in Android Settings → Security
- Or use ADB: `adb install --allow-downgrade human-alert.apk`

---

## 📊 Build Status

Check your build status at:
- **Terminal:** Watch the progress in real-time
- **Web:** https://expo.dev → Builds

---

## 🎉 Success!

Once installed, the Human Alert app will:
- ✅ Show the red Emergency button
- ✅ Track your GPS location
- ✅ Send/receive emergency alerts
- ✅ Display live map with directions
- ✅ Play notification sounds
- ✅ Vibrate on incoming alerts

---

## 📞 Support

If you encounter issues:
1. Check the [Expo Documentation](https://docs.expo.dev/build/setup/)
2. Visit the [EAS Build FAQ](https://docs.expo.dev/build-reference/faq/)

---

**Built with Expo SDK 54 | React Native 0.81**
