// firebase.js
import { initializeApp, getApps } from "firebase/app";
import { initializeAuth, getAuth, getReactNativePersistence } from "firebase/auth";
import { getFirestore, doc, setDoc } from "firebase/firestore";
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
let app;
let auth;
let db;

if (getApps().length === 0) {
  app = initializeApp(firebaseConfig);
  auth = initializeAuth(app, {
    persistence: getReactNativePersistence(ReactNativeAsyncStorage)
  });
  db = getFirestore(app);
} else {
  app = getApps()[0];
  auth = getAuth(app);
  db = getFirestore(app);
}

// Helper function to add or update user details in a "userDetails" collection
export const addUserDetails = async (userId, details) => {
  // doc(db, "userDetails", userId) will create or overwrite the document
  // with the userId as the doc ID in the "userDetails" collection.
  const docRef = doc(db, "userDetails", userId);
  await setDoc(docRef, details, { merge: true });
};

export { db };
export default auth;
