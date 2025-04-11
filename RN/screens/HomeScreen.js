// screens/homescreen.js

import React, { useContext, useState, useEffect, useCallback } from 'react';
import { View, Platform, StyleSheet, SafeAreaView, StatusBar, Image, Alert, ScrollView } from 'react-native';
import {
  Text, Button, Surface, useTheme, IconButton, Appbar,
  Menu, Portal, Dialog, TextInput, Avatar, ProgressBar
} from 'react-native-paper';
import * as ImagePicker from 'expo-image-picker';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import { ThemeContext } from '../context/ThemeContext';
import { signOut, onAuthStateChanged, updateProfile } from 'firebase/auth';
import auth from '../services/firebase';
import ErrorBoundary from '../ErrorBoundary';

const HomeScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const { isDarkTheme, toggleTheme } = useContext(ThemeContext);

  // State
  const [currentUser, setCurrentUser] = useState(null);
  const [displayName, setDisplayName] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [profileImage, setProfileImage] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [profileMenuVisible, setProfileMenuVisible] = useState(false);
  const [showEditProfile, setShowEditProfile] = useState(false);
  const [showErrorDialog, setShowErrorDialog] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [hasDetails, setHasDetails] = useState(false);
  const [prescription, setPrescription] = useState('');

  // Auth monitoring
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setCurrentUser(user);
      if (user) {
        setDisplayName(user.displayName || '');
        setProfileImage(user.photoURL || null);
      } else {
        navigation.replace('Login');
      }
    });
    return unsubscribe;
  }, [navigation]);

  // Fetch user details
  const fetchDetails = useCallback(() => {
    if (currentUser?.email) {
      fetch(`http://192.168.215.143:5000/getDetails?email=${currentUser.email}`)
        .then(res => res.json())
        .then(data => {
          const patient = data.details?.patient;
          setHasDetails(!!(patient?.firstName || patient?.lastName || patient?.gender || patient?.dob));
        })
        .catch(error => {
          console.error('Error fetching user details:', error);
          setHasDetails(false);
        });
    }
  }, [currentUser]);

  useFocusEffect(useCallback(() => { fetchDetails(); }, [fetchDetails]));

  // Image handling
  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });
      if (!result.canceled && result.assets?.[0]?.uri) {
        setSelectedImage(result.assets[0].uri);
      }
    } catch (error) {
      showError('Failed to pick image. Please try again.');
    }
  };

  const pickProfileImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });
      if (!result.canceled && result.assets?.[0]?.uri) {
        setProfileImage(result.assets[0].uri);
      }
    } catch (error) {
      showError('Failed to pick profile image. Please try again.');
    }
  };

  // User actions
  const handleSignOut = async () => {
    try {
      setIsLoading(true);
      await signOut(auth);
      setProfileMenuVisible(false);
    } catch (error) {
      showError('Failed to sign out. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdateProfile = async () => {
    try {
      setIsUpdating(true);
      if (!currentUser) throw new Error('No user logged in');
      
      simulateProgress();
      await updateProfile(currentUser, { 
        displayName: displayName,
        photoURL: profileImage 
      });
      
      setUploadProgress(1);
      setShowEditProfile(false);
      showError('Profile updated successfully!');
    } catch (error) {
      showError('Failed to update profile. Please try again.');
    } finally {
      setIsUpdating(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) {
      setErrorMessage('Please select an image first.');
      setShowErrorDialog(true);
      return;
    }
  
    try {
      setIsLoading(true);
      setUploadProgress(0.1);
      
      const formData = new FormData();
      formData.append('photo', {
        uri: selectedImage,
        type: 'image/jpeg',
        name: 'upload.jpg'
      });
      formData.append('prescription', prescription);
      
      const res = await fetch("http://192.168.215.143:5000/upload", {
        method: "POST",
        headers: {
          'Accept': 'application/json',
        },
        body: formData,
      });
  
      setUploadProgress(0.8);
      
      if (!res.ok) {
        const errorText = await res.text();
        console.error('Server error:', errorText, 'Status:', res.status);
        throw new Error(`Server error: ${res.status}`);
      }
      
      const result = await res.json();
      setUploadProgress(1);
      
      navigation.navigate('ResultScreen', {
        diagnosis: result.diagnosis,
        image_id: result.image_id
      });
    } catch (error) {
      console.error('Upload error:', error);
      setErrorMessage(`Failed to upload image. ${error.message || 'Please try again.'}`);
      setShowErrorDialog(true);
    } finally {
      setIsLoading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  // Helper
  const showError = (message) => {
    setErrorMessage(message);
    setShowErrorDialog(true);
  };

  const simulateProgress = () => {
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += 0.1;
      setUploadProgress(Math.min(progress, 0.9));
      if (progress >= 0.9) clearInterval(progressInterval);
    }, 100);
    return progressInterval;
  };

  return (
    <ErrorBoundary>
      <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <StatusBar barStyle={isDarkTheme ? 'light-content' : 'dark-content'} />
        
        <Appbar.Header style={styles.header} mode="center-aligned">
          <Appbar.Content title="Skin Lesion Classifier" />
          {/* Theme toggle icon */}
          <IconButton
            icon={() => <Feather name={isDarkTheme ? 'sun' : 'moon'} size={24} color={theme.colors.primary} />}
            onPress={toggleTheme}
          />
          {/* New info icon to view appointments */}
          <IconButton
            icon="information"
            size={24}
            color={theme.colors.primary}
            onPress={() => navigation.navigate('AppointmentDetails')}
          />
          <Menu
            visible={profileMenuVisible}
            onDismiss={() => setProfileMenuVisible(false)}
            anchor={<IconButton icon="account-circle" size={24} onPress={() => setProfileMenuVisible(true)} />}
          >
            <Menu.Item title={currentUser?.displayName || 'User'} description={currentUser?.email} />
            <Menu.Item
              leadingIcon="account-edit"
              onPress={() => {
                setShowEditProfile(true);
                setProfileMenuVisible(false);
              }}
              title="Edit Profile"
            />
            <Menu.Item leadingIcon="logout" onPress={handleSignOut} title="Sign Out" />
          </Menu>
        </Appbar.Header>

        <ScrollView style={styles.scrollView}>
          <View style={styles.mainContent}>
            <Surface style={styles.contentContainer} elevation={2}>
              <View style={styles.welcomeSection}>
                <MaterialCommunityIcons name="medical-bag" size={40} color={theme.colors.primary} />
                <Text variant="headlineSmall" style={styles.title}>AI-Powered Diagnosis</Text>
                <Text variant="bodyLarge" style={styles.subtitle}>
                  Upload your skin image for instant analysis and professional insights
                </Text>
              </View>
              
              <TextInput
                label="Medical Prescription"
                value={prescription}
                onChangeText={setPrescription}
                mode="outlined"
                multiline
                numberOfLines={4}
                style={styles.prescriptionInput}
              />
              
              {!hasDetails && (
                <View style={styles.addDetailsContainer}>
                  <Button
                    mode="contained"
                    onPress={() => navigation.navigate('Details')}
                    style={styles.addDetailsButton}
                  >
                    Add Details
                  </Button>
                </View>
              )}
              
              <Surface style={styles.imageSection} elevation={1}>
                {selectedImage ? (
                  <View style={styles.selectedImageContainer}>
                    <Image source={{ uri: selectedImage }} style={styles.image} resizeMode="cover" />
                    <IconButton
                      icon="close"
                      size={24}
                      onPress={() => setSelectedImage(null)}
                      style={styles.clearButton}
                    />
                  </View>
                ) : (
                  <View style={[styles.placeholderContainer, { borderColor: theme.colors.primary }]}>
                    <MaterialCommunityIcons name="image-plus" size={40} color={theme.colors.primary} />
                    <Text variant="bodyMedium" style={styles.uploadText}>Tap to select an image</Text>
                  </View>
                )}
              </Surface>
              
              {uploadProgress > 0 && (
                <ProgressBar progress={uploadProgress} color={theme.colors.primary} style={styles.progressBar} />
              )}
              
              <View style={styles.buttonContainer}>
                <Button
                  mode="outlined"
                  onPress={pickImage}
                  style={styles.button}
                  icon="image-multiple"
                  loading={isLoading}
                  disabled={isLoading}
                >
                  Choose from Gallery
                </Button>
                <Button
                  mode="contained"
                  onPress={handleUpload}
                  style={styles.button}
                  icon="upload"
                  loading={isLoading}
                  disabled={!selectedImage || isLoading}
                >
                  Analyze Image
                </Button>
              </View>
            </Surface>
          </View>
        </ScrollView>

        <Portal>
          <Dialog visible={showErrorDialog} onDismiss={() => setShowErrorDialog(false)}>
            <Dialog.Title>
              {errorMessage.toLowerCase().includes('error') ? 'Error' : 'Notice'}
            </Dialog.Title>
            <Dialog.Content>
              <Text variant="bodyMedium">{errorMessage}</Text>
            </Dialog.Content>
            <Dialog.Actions>
              <Button onPress={() => setShowErrorDialog(false)}>OK</Button>
            </Dialog.Actions>
          </Dialog>
          <Dialog visible={showEditProfile} onDismiss={() => setShowEditProfile(false)}>
            <Dialog.Title>Edit Profile</Dialog.Title>
            <Dialog.Content>
              <View style={styles.profileImageContainer}>
                <Avatar.Image
                  size={80}
                  source={profileImage ? { uri: profileImage } : require('../assets/adaptive-icon.png')}
                  style={styles.profileAvatar}
                />
                <IconButton
                  icon="camera"
                  size={24}
                  onPress={pickProfileImage}
                  style={styles.cameraButton}
                />
              </View>
              <TextInput
                label="Display Name"
                value={displayName}
                onChangeText={setDisplayName}
                style={styles.input}
                disabled={isUpdating}
              />
              <TextInput
                label="Phone Number"
                value={phoneNumber}
                onChangeText={setPhoneNumber}
                keyboardType="phone-pad"
                style={styles.input}
                disabled={isUpdating}
              />
              {uploadProgress > 0 && (
                <ProgressBar progress={uploadProgress} color={theme.colors.primary} style={styles.progressBar} />
              )}
            </Dialog.Content>
            <Dialog.Actions>
              <Button onPress={() => setShowEditProfile(false)} disabled={isUpdating}>Cancel</Button>
              <Button onPress={handleUpdateProfile} loading={isUpdating} disabled={isUpdating}>Save</Button>
            </Dialog.Actions>
          </Dialog>
        </Portal>
      </SafeAreaView>
    </ErrorBoundary>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  scrollView: { flex: 1 },
  header: {
    elevation: 4,
    shadowColor: 'rgba(0, 0, 0, 0.3)',
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 2,
  },
  mainContent: { flex: 1, padding: 16 },
  contentContainer: { padding: 20, borderRadius: 15, marginBottom: 16 },
  welcomeSection: { alignItems: 'center', marginBottom: 20 },
  title: { textAlign: 'center', marginVertical: 12, fontWeight: 'bold' },
  subtitle: { textAlign: 'center', opacity: 0.7, paddingHorizontal: 20 },
  prescriptionInput: { marginVertical: 16 },
  addDetailsContainer: { marginBottom: 20, alignItems: 'center' },
  addDetailsButton: { borderRadius: 8, paddingVertical: 6 },
  imageSection: { marginVertical: 16, borderRadius: 12, overflow: 'hidden' },
  selectedImageContainer: { position: 'relative' },
  image: { width: '100%', height: 250, borderRadius: 12 },
  clearButton: { position: 'absolute', top: 8, right: 8, backgroundColor: 'rgba(0, 0, 0, 0.5)' },
  placeholderContainer: { height: 250, borderWidth: 2, borderStyle: 'dashed', borderRadius: 12, justifyContent: 'center', alignItems: 'center' },
  uploadText: { marginTop: 12, opacity: 0.7 },
  buttonContainer: { gap: 12, marginTop: 16 },
  button: { borderRadius: 8, paddingVertical: 6 },
  progressBar: { marginVertical: 10, height: 4, borderRadius: 2 },
  profileImageContainer: { alignItems: 'center', marginVertical: 16, position: 'relative' },
  profileAvatar: { backgroundColor: '#e1e1e1' },
  input: { marginBottom: 12 },
  cameraButton: { backgroundColor: 'rgba(0, 0, 0, 0.3)', margin: 8 },
});

export default HomeScreen;
