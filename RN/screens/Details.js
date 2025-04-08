// screens/Details.js
import React, { useState, useContext } from 'react';
import { View, StyleSheet, SafeAreaView } from 'react-native';
import { 
  Text, 
  TextInput, 
  Button, 
  useTheme, 
  IconButton, 
  Appbar, 
  Snackbar
} from 'react-native-paper';
import { useNavigation } from '@react-navigation/native';
import { ThemeContext } from '../context/ThemeContext';
import { Feather } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

const DetailsScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const { isDarkTheme, toggleTheme } = useContext(ThemeContext);

  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName]   = useState('');
  const [gender, setGender]       = useState('');
  const [dob, setDob]             = useState('');
  const [error, setError]         = useState('');
  const [snackbarVisible, setSnackbarVisible] = useState(false);

  const handleSubmit = async () => {
    setError('');
    if (!firstName.trim() || !lastName.trim()) {
      setError('First Name and Last Name are required.');
      return;
    }

    // Retrieve the logged-in user's email from AsyncStorage
    const userDataString = await AsyncStorage.getItem('userData');
    const userData = userDataString ? JSON.parse(userDataString) : null;
    if (!userData || !userData.email) {
      setError('User not found. Please login again.');
      return;
    }

    // Save the details to the backend
    try {
      const response = await fetch('http://localhost:5000/details', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: userData.email,
          firstName,
          lastName,
          gender,
          dob,
        }),
      });
      const result = await response.json();
      if (response.ok) {
        setSnackbarVisible(true);
      } else {
        setError(result.error || 'Failed to update details');
      }
    } catch (err) {
      setError('Failed to update details');
    }
  };

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
        >
          Submit
        </Button>

        <Snackbar
          visible={snackbarVisible}
          onDismiss={() => {
            setSnackbarVisible(false);
            navigation.navigate('Home');
          }}
          duration={3000}
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
    paddingVertical: 6,
    borderRadius: 8,
  },
});

export default DetailsScreen;
