import { initializeApp, getApps } from "firebase/app";
import { initializeAuth, getAuth ,getReactNativePersistence } from "firebase/auth";
import ReactNativeAsyncStorage from '@react-native-async-storage/async-storage';

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyB2qaAey11oSffLIe7ZO-rQCZYdAM7Vd5E",
  authDomain: "rn-team2.firebaseapp.com",
  projectId: "rn-team2",
  storageBucket: "rn-team2.firebasestorage.app",
  messagingSenderId: "1049382548093",
  appId: "1:1049382548093:web:451e683884fe5489be7714"
};

// Initialize Firebase
let auth;
if (getApps().length === 0) {
    const app = initializeApp(firebaseConfig);
    // auth = initializeAuth(app, {
    //     persistence: getReactNativePersistence(ReactNativeAsyncStorage)
    //   });
    auth = initializeAuth(app, { 
        storage: ReactNativeAsyncStorage 
    });
} else {
    auth = getAuth();
}

export default auth;
