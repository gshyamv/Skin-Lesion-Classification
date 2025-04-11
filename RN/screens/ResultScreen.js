// screens/ResultScreen.js
import React, { useContext, useEffect, useState } from 'react';
import { View, StyleSheet, SafeAreaView, ScrollView, Alert } from 'react-native';
import { Text, Button, useTheme, IconButton, Appbar, Surface, Divider } from 'react-native-paper';
import { useNavigation, useRoute } from '@react-navigation/native';
import { ThemeContext } from '../context/ThemeContext';
import { Feather } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

const ResultScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const route = useRoute();
  const { isDarkTheme, toggleTheme } = useContext(ThemeContext);
  const { diagnosis, image_id } = route.params || {};
  const [city, setCity] = useState(route.params?.city);

  useEffect(() => {
    if (!city) {
      AsyncStorage.getItem('userData')
        .then((userData) => {
          if (userData) {
            const { email } = JSON.parse(userData);
            fetch(`http://192.168.215.143:5000/getDetails?email=${encodeURIComponent(email)}`)
              .then(response => response.json())
              .then(data => {
                if (data.details && data.details.contact && data.details.contact.city) {
                  setCity(data.details.contact.city);
                } else {
                  Alert.alert("Error", "City not found in user details.");
                }
              })
              .catch(err => {
                console.error("Error fetching user details:", err);
              });
          } else {
            console.error("User data not found in AsyncStorage");
          }
        })
        .catch(err => {
          console.error("Error reading user data:", err);
        });
    }
  }, [city]);

  const handleGoHome = () => {
    navigation.navigate('Home');
  };

  const handleShareResults = () => {
    // Implementation for sharing functionality.
  };

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
      <Appbar.Header style={styles.header} mode="center-aligned">
        <Appbar.BackAction onPress={() => navigation.goBack()} />
        <Appbar.Content title="Analysis Results" />
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
        <Appbar.Action
          icon="stethoscope"
          onPress={() => {
            if (city) {
              navigation.navigate('DoctorList', { city: city });
            } else {
              Alert.alert("City not set", "Patient city is not defined.");
            }
          }}
        />
      </Appbar.Header>

      <ScrollView style={styles.scrollView}>
        <View style={styles.container}>
          <Surface style={styles.resultContainer} elevation={2}>
            <View style={styles.headerSection}>
              <Feather name="clipboard" size={36} color={theme.colors.primary} />
              <Text variant="headlineSmall" style={styles.title}>
                Diagnostic Results
              </Text>
              <Text variant="bodyMedium" style={styles.subtitle}>
                AI-Powered Analysis of Your Skin Lesion
              </Text>
            </View>

            <Divider style={styles.divider} />

            <View style={styles.resultContent}>
              {diagnosis ? (
                <View style={styles.diagnosisContainer}>
                  <Text style={styles.label}>Diagnosis:</Text>
                  <Text style={styles.diagnosisText}>
                    {diagnosis}
                  </Text>
                </View>
              ) : (
                <Text style={styles.errorText}>
                  No diagnosis information available. Please try again.
                </Text>
              )}
            </View>

            <Divider style={styles.divider} />

            <Text style={styles.disclaimer}>
              IMPORTANT: This analysis is provided as a preliminary assessment only and should not replace 
              professional medical consultation. Please consult with a healthcare provider for a definitive diagnosis.
            </Text>
          </Surface>

          <View style={styles.actionButtons}>
            <Button
              mode="contained"
              onPress={handleGoHome}
              style={styles.button}
              icon="home"
            >
              Return Home
            </Button>
            <Button
              mode="outlined"
              onPress={handleShareResults}
              style={styles.button}
              icon="share-variant"
            >
              Share Results
            </Button>
          </View>
        </View>
      </ScrollView>
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
  scrollView: {
    flex: 1,
  },
  container: {
    flex: 1,
    padding: 16,
  },
  resultContainer: {
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
  },
  headerSection: {
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    textAlign: 'center',
    marginVertical: 8,
    fontWeight: 'bold',
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.7,
  },
  divider: {
    marginVertical: 16,
  },
  resultContent: {
    paddingVertical: 10,
  },
  diagnosisContainer: {
    marginBottom: 16,
  },
  label: {
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 8,
  },
  diagnosisText: {
    fontSize: 15,
    lineHeight: 22,
  },
  errorText: {
    color: 'red',
    textAlign: 'center',
    marginVertical: 16,
  },
  disclaimer: {
    fontStyle: 'italic',
    fontSize: 12,
    opacity: 0.7,
    textAlign: 'center',
  },
  actionButtons: {
    flexDirection: 'column',
    gap: 12,
  },
  button: {
    borderRadius: 8,
    paddingVertical: 6,
  },
});

export default ResultScreen;
