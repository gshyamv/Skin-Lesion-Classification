import React, { useContext, useState, useEffect, useCallback } from 'react';
import { config } from '../Utils/config';
import { uploadImageToMongo, getImagesFromMongo } from '../Utils/mongoUtils';
import {
  View,
  Platform,
  StyleSheet,
  SafeAreaView,
  StatusBar,
  Image,
  Alert,
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
  Avatar,
  FAB,
  ProgressBar,
} from 'react-native-paper';
import * as ImagePicker from 'expo-image-picker';
import { Camera } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library';
import { useNavigation } from '@react-navigation/native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import { ThemeContext } from '../context/ThemeContext';
import { signOut, onAuthStateChanged, updateProfile } from 'firebase/auth';
import auth from '../services/firebase';

// ErrorBoundary component for handling runtime errors gracefully
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
          <Button
            mode="contained"
            onPress={() => this.setState({ hasError: false })}
            style={styles.errorButton}
          >
            Try Again
          </Button>
        </View>
      );
    }
    return this.props.children;
  }
}

// Main HomeScreen component
const HomeScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const { isDarkTheme, toggleTheme } = useContext(ThemeContext);

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

  const [cameraPermission, setCameraPermission] = useState(null);
  const [mediaLibraryPermission, setMediaLibraryPermission] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [cameraType, setCameraType] = useState(() => Camera.Constants?.Type.back);
  const [cameraRef, setCameraRef] = useState(null);
  const [flash, setFlash] = useState(() => Camera.Constants?.FlashMode.off);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isCameraInitialized, setIsCameraInitialized] = useState(false);

  useEffect(() => {
    let mounted = true;

    const requestPermissions = async () => {
      try {
        // Request camera permission with proper error handling
        const { status: cameraStatus } = await Camera.requestCameraPermissionsAsync();
        if (mounted) {
          setCameraPermission(cameraStatus === 'granted');
          
          // Only proceed with media library permission if camera permission is granted
          if (cameraStatus === 'granted') {
            const { status: mediaStatus } = await MediaLibrary.requestPermissionsAsync();
            setMediaLibraryPermission(mediaStatus === 'granted');
          }
        }
      } catch (error) {
        console.error('Permission request error:', error);
        if (mounted) {
          setErrorMessage('Failed to initialize camera. Please check permissions and try again.');
          setShowErrorDialog(true);
        }
      }
    };

    requestPermissions();
    return () => {
      mounted = false;
    };
  }, []);

  const initializeCamera = useCallback(async (ref) => {
    if (!ref) return;
    
    try {
      setCameraRef(ref);
      // Wait for camera to be ready
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setIsCameraInitialized(true);
    } catch (error) {
      console.error('Camera initialization error:', error);
      setErrorMessage('Failed to initialize camera. Please try again.');
      setShowErrorDialog(true);
    }
  }, []);

  // Monitor authentication state
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

  const handleCameraReady = useCallback(() => {
    setIsCameraReady(true);
  }, []);

  const handleTakePhoto = async () => {
    if (!cameraRef || !isCameraReady || !isCameraInitialized) {
      setErrorMessage('Please wait for camera to initialize completely.');
      setShowErrorDialog(true);
      return;
    }

    try {
      setIsLoading(true);

      // Add a small delay to ensure camera is fully ready
      await new Promise(resolve => setTimeout(resolve, 300));

      const photo = await cameraRef.takePictureAsync({
        quality: 0.8,
        base64: false,
        exif: false,
        skipProcessing: Platform.OS === 'android',
        // Ensure proper formatting for web platforms
        format: Platform.OS === 'web' ? 'jpeg' : 'auto'
      });

      if (!photo || !photo.uri) {
        throw new Error('Failed to capture photo');
      }

      // Save to media library if permission granted and not on web
      if (mediaLibraryPermission === 'granted' && Platform.OS !== 'web') {
        await MediaLibrary.saveToLibraryAsync(photo.uri);
      }

      setSelectedImage(photo.uri);
      setShowCamera(false);
    } catch (error) {
      console.error('Photo capture error:', error);
      setErrorMessage(
        Platform.OS === 'web'
          ? 'Please ensure camera permissions are granted in your browser settings.'
          : 'Failed to capture photo. Please try again.'
      );
      setShowErrorDialog(true);
    } finally {
      setIsLoading(false);
    }
  };

  // Camera control handlers
  const toggleFlash = () => {
    setFlash(currentFlash =>
      currentFlash === Camera.Constants?.FlashMode.off
        ? Camera.Constants?.FlashMode.on
        : Camera.Constants?.FlashMode.off
    );
  };

  const toggleCameraType = () => {
    setCameraType(currentType =>
      currentType === Camera.Constants?.Type.back
        ? Camera.Constants?.Type.front
        : Camera.Constants?.Type.back
    );
  };

  // Image picker handler
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

  // Profile image picker
  const pickProfileImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 1,
      });

      if (!result.canceled && result.assets?.[0]?.uri) {
        setProfileImage(result.assets[0].uri);
      }
    } catch (error) {
      setErrorMessage('Failed to pick profile image. Please try again.');
      setShowErrorDialog(true);
    }
  };

  // Authentication handlers
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

  // Profile update handler
  const handleUpdateProfile = async () => {
    try {
      setIsUpdating(true);

      if (!currentUser) {
        throw new Error('No user logged in');
      }

      // Simulate upload progress
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += 0.1;
        setUploadProgress(Math.min(progress, 0.9));
      }, 100);

      // Update profile
      await updateProfile(currentUser, {
        photoURL: profileImage,
      });

      clearInterval(progressInterval);
      setUploadProgress(1);

      setShowEditProfile(false);
      setErrorMessage('Profile updated successfully!');
      setShowErrorDialog(true);
    } catch (error) {
      setErrorMessage('Failed to update profile. Please try again.');
      setShowErrorDialog(true);
    } finally {
      setIsUpdating(false);
      setUploadProgress(0);
    }
  };

  // Image upload and analysis handler (updated to convert URI to Blob)
  const handleUpload = async () => {
    if (!selectedImage) {
      setErrorMessage('Please select an image first.');
      setShowErrorDialog(true);
      return;
    }
  
    try {
      setIsLoading(true);
      setUploadProgress(0.1);
      
      // Convert the image URI to a blob
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      
      const formData = new FormData();
      formData.append("photo", blob, "image.jpg");
  
      const res = await fetch("http://192.168.222.143:5000/upload", {
        method: "POST",
        body: formData,
      });
  
      const result = await res.json();
      if (res.ok) {
        Alert.alert("Success", "Image uploaded successfully!");
      } else {
        Alert.alert("Error", result.error);
      }
      setUploadProgress(1);
    } catch (error) {
      console.error('Upload error:', error);
      setErrorMessage('Failed to upload image. Please try again.');
      setShowErrorDialog(true);
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
    }
  };
      
  // Camera component
  const CameraComponent = () => {
    if (!cameraPermission) {
      return (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>
            Camera permission is required. Please enable it in your settings.
          </Text>
          <Button
            mode="contained"
            onPress={() => setShowCamera(false)}
            style={styles.errorButton}
          >
            Go Back
          </Button>
        </View>
      );
    }
  
    return (
      <View style={styles.cameraContainer}>
        <StatusBar translucent backgroundColor="transparent" barStyle="light-content" />
        <Camera
          ref={initializeCamera}
          style={styles.camera}
          type={cameraType}
          flashMode={flash}
          onCameraReady={handleCameraReady}
          useCamera2Api={Platform.OS === 'android'}
          ratio="16:9"
          onMountError={(error) => {
            console.error('Camera mount error:', error);
            setErrorMessage('Failed to start camera. Please try again.');
            setShowErrorDialog(true);
          }}
        >
          <SafeAreaView style={styles.cameraControlsContainer}>
            <View style={styles.cameraControls}>
              <IconButton
                icon="close"
                size={30}
                iconColor="white"
                onPress={() => {
                  setShowCamera(false);
                  setIsCameraInitialized(false);
                }}
                style={styles.cameraButton}
              />
              {Platform.OS !== 'web' && (
                <IconButton
                  icon={flash === Camera.Constants.FlashMode.off ? 'flash-off' : 'flash'}
                  size={30}
                  iconColor="white"
                  onPress={toggleFlash}
                  style={styles.cameraButton}
                />
              )}
              <IconButton
                icon="camera-flip"
                size={30}
                iconColor="white"
                onPress={toggleCameraType}
                style={styles.cameraButton}
              />
            </View>
          </SafeAreaView>
          
          <FAB
            icon="camera"
            style={[
              styles.captureButton,
              (!isCameraReady || !isCameraInitialized) && styles.disabledButton
            ]}
            onPress={handleTakePhoto}
            disabled={!isCameraReady || !isCameraInitialized || isLoading}
            loading={isLoading}
            color="white"
          />
        </Camera>
      </View>
    );
  };
  
  return (
    <ErrorBoundary>
      {showCamera ? (
        <CameraComponent />
      ) : (
        <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
          <StatusBar barStyle={isDarkTheme ? 'light-content' : 'dark-content'} />
          <Appbar.Header style={styles.header} mode="center-aligned">
            <Appbar.Content title="Skin Lesion Classifier" />
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
                leadingIcon="account-edit"
                onPress={() => {
                  setShowEditProfile(true);
                  setProfileMenuVisible(false);
                }}
                title="Edit Profile"
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
                    <Image
                      source={{ uri: selectedImage }}
                      style={styles.image}
                      resizeMode="cover"
                    />
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
                      Tap to select or capture an image
                    </Text>
                  </View>
                )}
              </Surface>
  
              {uploadProgress > 0 && (
                <ProgressBar
                  progress={uploadProgress}
                  color={theme.colors.primary}
                  style={styles.progressBar}
                />
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
                  mode="outlined"
                  onPress={() => setShowCamera(true)}
                  style={styles.button}
                  icon="camera"
                  loading={isLoading}
                  disabled={isLoading || !cameraPermission}
                >
                  Take Photo
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
  
          {/* Portal for Dialogs */}
          <Portal>
            {/* Error Dialog */}
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
  
            {/* Edit Profile Dialog */}
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
                  <ProgressBar
                    progress={uploadProgress}
                    color={theme.colors.primary}
                    style={styles.progressBar}
                  />
                )}
              </Dialog.Content>
              <Dialog.Actions>
                <Button onPress={() => setShowEditProfile(false)} disabled={isUpdating}>
                  Cancel
                </Button>
                <Button onPress={handleUpdateProfile} loading={isUpdating} disabled={isUpdating}>
                  Save
                </Button>
              </Dialog.Actions>
            </Dialog>
          </Portal>
        </SafeAreaView>
      )}
    </ErrorBoundary>
  );
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    marginVertical: 16,
    textAlign: 'center',
    fontSize: 16,
  },
  errorButton: {
    marginTop: 12,
  },
  header: {
    elevation: 4,
    shadowColor: 'rgba(0, 0, 0, 0.3)',
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 2,
  },
  mainContent: {
    flex: 1,
    padding: 16,
  },
  contentContainer: {
    padding: 20,
    borderRadius: 15,
    height: '100%',
  },
  welcomeSection: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    textAlign: 'center',
    marginVertical: 12,
    fontWeight: 'bold',
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.7,
    paddingHorizontal: 20,
  },
  imageSection: {
    marginVertical: 20,
    borderRadius: 12,
    overflow: 'hidden',
  },
  selectedImageContainer: {
    position: 'relative',
  },
  image: {
    width: '100%',
    height: 300,
    borderRadius: 12,
  },
  clearButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  placeholderContainer: {
    height: 300,
    borderWidth: 2,
    borderStyle: 'dashed',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  uploadText: {
    marginTop: 12,
    opacity: 0.7,
  },
  buttonContainer: {
    gap: 12,
    marginTop: 20,
  },
  button: {
    borderRadius: 8,
    paddingVertical: 6,
  },
  progressBar: {
    marginVertical: 10,
    height: 4,
    borderRadius: 2,
  },
  // Camera styles
  cameraContainer: {
    flex: 1,
    backgroundColor: 'black',
  },
  camera: {
    flex: 1,
    aspectRatio: Platform.OS === 'web' ? 16 / 9 : undefined,
  },
  cameraControlsContainer: {
    flex: 1,
    justifyContent: 'space-between',
    backgroundColor: 'transparent',
    flexDirection: 'row',
  },
  cameraControls: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
    paddingTop: Platform.OS === 'ios' ? 40 : 20,
  },
  cameraButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    margin: 8,
  },
  captureButton: {
    position: 'absolute',
    bottom: Platform.OS === 'web' ? 20 : 40,
    alignSelf: 'center',
    backgroundColor: '#2196F3',
  },
  disabledButton: {
    opacity: 0.5,
  },
  // Profile styles
  profileImageContainer: {
    alignItems: 'center',
    marginVertical: 20,
    position: 'relative',
  },
  profileAvatar: {
    backgroundColor: '#e1e1e1',
  },
  input: {
    marginBottom: 16,
  },
  // Platform-specific shadow styles
  ...Platform.select({
    ios: {
      header: {
        shadowColor: 'rgba(0, 0, 0, 0.3)',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 2,
      },
    },
    android: {
      header: {
        elevation: 4,
      },
    },
  }),
});

export default HomeScreen;
