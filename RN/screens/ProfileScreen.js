// screens/ProfileScreen.js
import React, { useState, useEffect } from 'react';
import { View, StyleSheet, SafeAreaView, ScrollView } from 'react-native';
import { Text, Appbar, Avatar, useTheme, Button, ActivityIndicator } from 'react-native-paper';
import { useNavigation } from '@react-navigation/native';
import auth from '../services/firebase';

const ProfileScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  // Get the current user from Firebase auth
  const currentUser = auth.currentUser;

  useEffect(() => {
    const fetchProfile = async () => {
      if (!currentUser?.email) {
        setLoading(false);
        return;
      }
      try {
        const res = await fetch(`http://localhost:5000/getDetails?email=${currentUser.email}`);
        const data = await res.json();
        if (res.ok) {
          setProfile(data.details);
        } else {
          setProfile(null);
        }
      } catch (error) {
        console.error(error);
        setProfile(null);
      } finally {
        setLoading(false);
      }
    };

    fetchProfile();
  }, [currentUser]);

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Appbar.Header style={styles.header} mode="center-aligned">
        <Appbar.BackAction onPress={() => navigation.goBack()} />
        <Appbar.Content title="Profile" />
      </Appbar.Header>
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
        </View>
      ) : (
        <ScrollView contentContainerStyle={styles.content}>
          <Avatar.Image 
            size={100}
            source={ currentUser?.photoURL ? { uri: currentUser.photoURL } : require('../assets/adaptive-icon.png') }
          />
          <Text variant="headlineMedium" style={styles.name}>
            {profile?.name || currentUser?.displayName || 'User'}
          </Text>
          <Text variant="bodyMedium" style={styles.email}>
            {currentUser?.email}
          </Text>
          {profile?.details && Object.keys(profile.details).length > 0 && (
            <View style={styles.detailsContainer}>
              {profile.details.firstName && (
                <Text variant="bodyLarge">First Name: {profile.details.firstName}</Text>
              )}
              {profile.details.lastName && (
                <Text variant="bodyLarge">Last Name: {profile.details.lastName}</Text>
              )}
              {profile.details.gender && (
                <Text variant="bodyLarge">Gender: {profile.details.gender}</Text>
              )}
              {profile.details.dob && (
                <Text variant="bodyLarge">DOB: {profile.details.dob}</Text>
              )}
            </View>
          )}
          <Button mode="contained" onPress={() => navigation.goBack()} style={styles.backButton}>
            Back
          </Button>
        </ScrollView>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: { elevation: 4 },
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  content: { alignItems: 'center', padding: 20 },
  name: { marginTop: 10 },
  email: { marginBottom: 20 },
  detailsContainer: { marginVertical: 20 },
  backButton: { marginTop: 20 },
});

export default ProfileScreen;
