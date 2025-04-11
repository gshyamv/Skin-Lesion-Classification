// App.js
import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { Provider as PaperProvider, MD3DarkTheme, MD3LightTheme } from 'react-native-paper';
import AsyncStorage from '@react-native-async-storage/async-storage';
import RegisterScreen from './screens/rs1';
import HomeScreen from './screens/HomeScreen';
import MainScreen from './screens/MainScreen';
import LoginScreen from './screens/LoginScreen';
import ResultScreen from './screens/ResultScreen'; 
import DoctorListScreen from './screens/DoctorListScreens';
import BookAppointmentScreen from './screens/BookAppointmentScreen';
import AppointmentDetailsScreen from './screens/AppointmentDetailsScreen';  // Import the new screen
import { ThemeContext } from './context/ThemeContext';
import auth from './services/firebase';
import DetailsScreen from './screens/Details';
import { onAuthStateChanged } from 'firebase/auth';
import { ActivityIndicator, View } from 'react-native';

const Stack = createStackNavigator();

const lightTheme = {
  ...MD3LightTheme,
  colors: {
    ...MD3LightTheme.colors,
    background: '#f5f5f5',
    surface: '#ffffff',
    text: '#000000',
    primary: '#6200ee',
    error: '#B00020',
  },
};

const darkTheme = {
  ...MD3DarkTheme,
  colors: {
    ...MD3DarkTheme.colors,
    background: '#121212',
    surface: '#1e1e1e',
    text: '#ffffff',
    primary: '#bb86fc',
    error: '#CF6679',
  },
};

export default function App() {
  const [isDarkTheme, setIsDarkTheme] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const theme = isDarkTheme ? darkTheme : lightTheme;

  useEffect(() => {
    const loadThemePreference = async () => {
      try {
        const savedTheme = await AsyncStorage.getItem('isDarkTheme');
        if (savedTheme !== null) {
          setIsDarkTheme(JSON.parse(savedTheme));
        }
      } catch (error) {
        console.error('Error loading theme preference:', error);
      }
    };
    loadThemePreference();
  }, []);

  const toggleTheme = async () => {
    try {
      const newTheme = !isDarkTheme;
      setIsDarkTheme(newTheme);
      await AsyncStorage.setItem('isDarkTheme', JSON.stringify(newTheme));
    } catch (error) {
      console.error('Error saving theme preference:', error);
    }
  };

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setIsAuthenticated(!!user);
      setIsLoading(false);
      if (user) {
        try {
          await AsyncStorage.setItem('userData', JSON.stringify({
            uid: user.uid,
            email: user.email,
            displayName: user.displayName
          }));
        } catch (error) {
          console.error('Error storing user data:', error);
        }
      } else {
        try {
          await AsyncStorage.removeItem('userData');
        } catch (error) {
          console.error('Error removing user data:', error);
        }
      }
    });

    return () => unsubscribe();
  }, []);

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
      </View>
    );
  }

  return (
    <ThemeContext.Provider value={{ isDarkTheme, toggleTheme }}>
      <PaperProvider theme={theme}>
        <NavigationContainer>
          <Stack.Navigator
            initialRouteName={isAuthenticated ? "Home" : "Main"}
            screenOptions={{
              headerShown: false,
              gestureEnabled: true,
            }}
          >
            {isAuthenticated ? (
              <>
                <Stack.Screen name="Home" component={HomeScreen} />
                <Stack.Screen name="Details" component={DetailsScreen} />
                <Stack.Screen name="ResultScreen" component={ResultScreen} />
                <Stack.Screen name="DoctorList" component={DoctorListScreen} />
                <Stack.Screen name="BookAppointment" component={BookAppointmentScreen} />
                {/* New screen added to authenticated stack */}
                <Stack.Screen name="AppointmentDetails" component={AppointmentDetailsScreen} />
              </>
            ) : (
              <>
                <Stack.Screen name="Main" component={MainScreen} />
                <Stack.Screen name="Login" component={LoginScreen} />
                <Stack.Screen name="Register" component={RegisterScreen} />
                <Stack.Screen name="ResultScreen" component={ResultScreen} /> 
                <Stack.Screen name="DoctorList" component={DoctorListScreen} />
                <Stack.Screen name="BookAppointment" component={BookAppointmentScreen} />
                {/* Optionally include AppointmentDetails here if needed */}
                <Stack.Screen name="AppointmentDetails" component={AppointmentDetailsScreen} />
              </>
            )}
          </Stack.Navigator>
        </NavigationContainer>
      </PaperProvider>
    </ThemeContext.Provider>
  );
}
