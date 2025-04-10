// screens/Details.js
import React, { useState, useContext, useEffect } from 'react';
import { View, StyleSheet, SafeAreaView, BackHandler } from 'react-native';
import { 
  Text, 
  TextInput, 
  Button, 
  useTheme, 
  IconButton, 
  Appbar, 
  Snackbar,
  ActivityIndicator
} from 'react-native-paper';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { ThemeContext } from '../context/ThemeContext';
import { Feather } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

const DetailsScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const { isDarkTheme, toggleTheme } = useContext(ThemeContext);

  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [gender, setGender] = useState('');
  const [dob, setDob] = useState('');
  const [error, setError] = useState('');
  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [userEmail, setUserEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Handle back button to prevent accidental navigation away
  useFocusEffect(
    React.useCallback(() => {
      const onBackPress = () => {
        if (!isSubmitting) {
          navigation.navigate('Home');
          return true;
        }
        return false;
      };

      BackHandler.addEventListener('hardwareBackPress', onBackPress);
      return () => BackHandler.removeEventListener('hardwareBackPress', onBackPress);
    }, [isSubmitting, navigation])
  );

  // Get user data from AsyncStorage
  useEffect(() => {
    const getUserData = async () => {
      try {
        const userDataString = await AsyncStorage.getItem('userData');
        if (userDataString) {
          const userData = JSON.parse(userDataString);
          setUserEmail(userData.email);
          
          // Check if user already has details
          if (userData.email) {
            const response = await fetch(`http://localhost:5000/getDetails?email=${userData.email}`);
            const data = await response.json();
            
            if (data.details && data.details.details) {
              const userDetails = data.details.details;
              // Pre-fill the form if details exist
              setFirstName(userDetails.firstName || '');
              setLastName(userDetails.lastName || '');
              setGender(userDetails.gender || '');
              setDob(userDetails.dob || '');
            }
          }
        }
      } catch (error) {
        console.error('Error fetching user data:', error);
        setError('Failed to fetch user data');
      } finally {
        setIsLoading(false);
      }
    };
    
    getUserData();
  }, []);

  const handleSubmit = async () => {
    setError('');
    if (!firstName.trim() || !lastName.trim()) {
      setError('First Name and Last Name are required.');
      return;
    }

    if (!userEmail) {
      setError('User not found. Please login again.');
      return;
    }

    // Save the details to the backend
    try {
      setIsSubmitting(true);
      
      const response = await fetch('http://localhost:5000/details', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: userEmail,
          firstName,
          lastName,
          gender,
          dob,
        }),
      });
      
      const result = await response.json();
      if (response.ok) {
        setSnackbarVisible(true);
        // Wait for snackbar to be visible before navigating away
        setTimeout(() => {
          navigation.navigate('Home');
        }, 2000);
      } else {
        setError(result.error || 'Failed to update details');
      }
    } catch (err) {
      console.error('Error submitting details:', err);
      setError('Failed to update details. Please check your network connection.');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isLoading) {
    return (
      <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
        <Appbar.Header style={styles.header} mode="center-aligned">
          <Appbar.BackAction onPress={() => navigation.goBack()} />
          <Appbar.Content title="Your Details" />
        </Appbar.Header>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
          <Text style={styles.loadingText}>Loading your details...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
      <Appbar.Header style={styles.header} mode="center-aligned">
        <Appbar.BackAction onPress={() => navigation.goBack()} />
        <Appbar.Content title="Your Details" />
        <IconButton
          icon={() => (
            <Feather
              name={isDarkTheme ? 'sun' : 'moon'}
              size={24}
              color={theme.colors.primary}
            />
          )}
          onPress={toggleTheme}
        />
      </Appbar.Header>

      <View style={styles.container}>
        <Text variant="titleLarge" style={[styles.title, { color: theme.colors.primary }]}>
          Please Enter Your Details
        </Text>

        <TextInput
          label="First Name (Required)"
          value={firstName}
          onChangeText={setFirstName}
          mode="outlined"
          style={styles.input}
        />
        <TextInput
          label="Last Name (Required)"
          value={lastName}
          onChangeText={setLastName}
          mode="outlined"
          style={styles.input}
        />
        <TextInput
          label="Gender (Optional)"
          value={gender}
          onChangeText={setGender}
          mode="outlined"
          style={styles.input}
          placeholder="e.g., Male/Female/Other"
        />
        <TextInput
          label="Date of Birth (Optional)"
          value={dob}
          onChangeText={setDob}
          mode="outlined"
          style={styles.input}
          placeholder="YYYY-MM-DD"
        />

        {error ? (
          <Text style={[styles.errorText, { color: theme.colors.error }]}>
            {error}
          </Text>
        ) : null}

        <Button
          mode="contained"
          onPress={handleSubmit}
          style={styles.submitButton}
          disabled={isSubmitting}
          loading={isSubmitting}
        >
          {isSubmitting ? 'Submitting...' : 'Submit'}
        </Button>

        <Snackbar
          visible={snackbarVisible}
          onDismiss={() => {
            setSnackbarVisible(false);
          }}
          duration={2000}
        >
          Details submitted successfully!
        </Snackbar>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  header: {
    elevation: 4,
  },
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
  },
  title: {
    marginBottom: 30,
    textAlign: 'center',
    fontWeight: 'bold',
  },
  input: {
    marginBottom: 15,
  },
  errorText: {
    marginBottom: 10,
    textAlign: 'center',
  },
  submitButton: {
    marginTop: 10,
    paddingVertical: 8,
    borderRadius: 5,
  }
});

export default DetailsScreen;