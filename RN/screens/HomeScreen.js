// screens/HomeScreen.js
import React, { useContext, useState, useEffect } from 'react';
import { 
  View, 
  StyleSheet, 
  SafeAreaView, 
  StatusBar, 
  Image, 
  Alert 
} from 'react-native';
import {
  Text,
  Button,
  Surface,
  useTheme,
  IconButton,
  Appbar,
  Menu,
  Portal,
  Dialog,
  TextInput,
  ProgressBar,
} from 'react-native-paper';
import * as ImagePicker from 'expo-image-picker';
import { useNavigation } from '@react-navigation/native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import { ThemeContext } from '../context/ThemeContext';
import { signOut, onAuthStateChanged, updateProfile } from 'firebase/auth';
import auth from '../services/firebase';

class ErrorBoundary extends React.Component {
  constructor(props) {  
    super(props);
    this.state = { hasError: false, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.errorContainer}>
          <MaterialCommunityIcons name="alert-circle" size={48} color="#FF6B6B" />
          <Text style={styles.errorText}>Something went wrong.</Text>
          <Button mode="contained" onPress={() => this.setState({ hasError: false })} style={styles.errorButton}>
            Try Again
          </Button>
        </View>
      );
    }
    return this.props.children;
  }
}

const HomeScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const { isDarkTheme, toggleTheme } = useContext(ThemeContext);
  const [currentUser, setCurrentUser] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [profileMenuVisible, setProfileMenuVisible] = useState(false);
  const [showErrorDialog, setShowErrorDialog] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [hasDetails, setHasDetails] = useState(false);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setCurrentUser(user);
      if (!user) {
        navigation.replace('Login');
      }
    });
    return unsubscribe;
  }, [navigation]);

  // Fetch user details from backend to check if they already exist
  useEffect(() => {
    if (currentUser && currentUser.email) {
      fetch(`http://localhost:5000/getDetails?email=${currentUser.email}`)
        .then(res => res.json())
        .then(data => {
          if (data.details) {
            setHasDetails(true);
          }
        })
        .catch(err => console.error(err));
    }
  }, [currentUser]);

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });

      if (!result.canceled && result.assets?.[0]?.uri) {
        setSelectedImage(result.assets[0].uri);
      }
    } catch (error) {
      setErrorMessage('Failed to pick image. Please try again.');
      setShowErrorDialog(true);
    }
  };

  const handleSignOut = async () => {
    try {
      setIsLoading(true);
      await signOut(auth);
      setProfileMenuVisible(false);
    } catch (error) {
      setErrorMessage('Failed to sign out. Please try again.');
      setShowErrorDialog(true);
    } finally {
      setIsLoading(false);
    }
  };

  // Updated: Navigate to the ResultScreen after analysis
  const handleUpload = async () => {
    if (!selectedImage) {
      setErrorMessage('Please select an image first.');
      setShowErrorDialog(true);
      return;
    }
  
    try {
      setIsLoading(true);
      setUploadProgress(0.1);
      
      // Simulate image upload and analysis (replace with actual API calls as needed)
      // For example: upload image to the backend and receive analysis result.
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      const formData = new FormData();
      formData.append("photo", blob, "image.jpg");
  
      // Here we call your API endpoint to upload the image.
      const res = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });
  
      const result = await res.json();
      if (!res.ok) {
        Alert.alert("Error", result.error);
        setUploadProgress(0);
        return;
      }
      
      // Simulate progress update
      setUploadProgress(1);
      
      // Navigate to the final ResultScreen with the required parameters.
      navigation.navigate('Result', {
        imageUri: selectedImage,      // URI of the uploaded image
        cancerType: "Melanoma",         // Replace with your classifier’s output
        malignancyScore: 0.75,          // A value between 0.0 (benign) and 1.0 (malignant)
      });
    } catch (error) {
      console.error('Upload error:', error);
      setErrorMessage('Failed to upload image. Please try again.');
      setShowErrorDialog(true);
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
    }
  };

  return (
    <ErrorBoundary>
      <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <StatusBar barStyle={isDarkTheme ? 'light-content' : 'dark-content'} />
        <Appbar.Header style={styles.header} mode="center-aligned">
          <Appbar.Content title="Skin Lesion Classifier" />
          <IconButton
            icon={() => (
              <Feather name={isDarkTheme ? 'sun' : 'moon'} size={24} color={theme.colors.primary} />
            )}
            onPress={toggleTheme}
          />
          <Menu
            visible={profileMenuVisible}
            onDismiss={() => setProfileMenuVisible(false)}
            anchor={
              <IconButton
                icon="account-circle"
                size={24}
                onPress={() => setProfileMenuVisible(true)}
              />
            }
          >
            <Menu.Item
              title={currentUser?.displayName || 'User'}
              description={currentUser?.email}
            />
            <Menu.Item
              leadingIcon="account"
              onPress={() => {
                navigation.navigate('Profile');
                setProfileMenuVisible(false);
              }}
              title="Profile"
            />
            <Menu.Item
              leadingIcon="logout"
              onPress={handleSignOut}
              title="Sign Out"
            />
          </Menu>
        </Appbar.Header>
  
        <View style={styles.mainContent}>
          <Surface style={styles.contentContainer} elevation={2}>
            <View style={styles.welcomeSection}>
              <MaterialCommunityIcons
                name="medical-bag"
                size={40}
                color={theme.colors.primary}
              />
              <Text variant="headlineSmall" style={styles.title}>
                AI-Powered Diagnosis
              </Text>
              <Text variant="bodyLarge" style={styles.subtitle}>
                Upload your skin image for instant analysis and professional insights
              </Text>
            </View>
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
                  <MaterialCommunityIcons
                    name="image-plus"
                    size={40}
                    color={theme.colors.primary}
                  />
                  <Text variant="bodyMedium" style={styles.uploadText}>
                    Click on (Choose from Gallery) button below to select an image
                  </Text>
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
  
        <Portal>
          <Dialog visible={showErrorDialog} onDismiss={() => setShowErrorDialog(false)}>
            <Dialog.Title>{errorMessage.toLowerCase().includes('error') ? 'Error' : 'Notice'}</Dialog.Title>
            <Dialog.Content>
              <Text variant="bodyMedium">{errorMessage}</Text>
            </Dialog.Content>
            <Dialog.Actions>
              <Button onPress={() => setShowErrorDialog(false)}>OK</Button>
            </Dialog.Actions>
          </Dialog>
        </Portal>
      </SafeAreaView>
    </ErrorBoundary>
  );
};

export default HomeScreen;

const styles = StyleSheet.create({
  container: { flex: 1 },
  errorContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  errorText: { marginVertical: 16, textAlign: 'center', fontSize: 16 },
  errorButton: { marginTop: 12 },
  header: { elevation: 4 },
  mainContent: { flex: 1, padding: 16 },
  contentContainer: { padding: 20, borderRadius: 15, height: '100%' },
  welcomeSection: { alignItems: 'center', marginBottom: 30 },
  title: { textAlign: 'center', marginVertical: 12, fontWeight: 'bold' },
  subtitle: { textAlign: 'center', opacity: 0.7, paddingHorizontal: 20 },
  imageSection: { marginVertical: 20, borderRadius: 12, overflow: 'hidden' },
  selectedImageContainer: { position: 'relative' },
  image: { width: '100%', height: 300, borderRadius: 12 },
  clearButton: { position: 'absolute', top: 8, right: 8, backgroundColor: 'rgba(0, 0, 0, 0.5)' },
  placeholderContainer: { height: 300, borderWidth: 2, borderStyle: 'dashed', borderRadius: 12, justifyContent: 'center', alignItems: 'center' },
  uploadText: { marginTop: 12, opacity: 0.7 },
  buttonContainer: { gap: 12, marginTop: 20 },
  button: { borderRadius: 8, paddingVertical: 6 },
  progressBar: { marginVertical: 10, height: 4, borderRadius: 2 },
});