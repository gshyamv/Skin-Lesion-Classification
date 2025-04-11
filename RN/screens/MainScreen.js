import React, { useContext } from 'react';
import { View, StyleSheet, SafeAreaView, StatusBar, ScrollView } from 'react-native';
import { Text, Button, Surface, useTheme, IconButton, Appbar } from 'react-native-paper';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { ThemeContext } from '../context/ThemeContext';

const MainScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const { isDarkTheme, toggleTheme } = useContext(ThemeContext);

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <StatusBar barStyle={theme.dark ? 'light-content' : 'dark-content'} />
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
      </Appbar.Header>

      <ScrollView style={styles.mainContent}>
        <Surface style={styles.contentContainer} elevation={2}>
          <View style={styles.welcomeSection}>
            <MaterialCommunityIcons 
              name="medical-bag" 
              size={40} 
              color={theme.colors.primary}
            />
            <Text variant="headlineSmall" style={styles.title}>
              Welcome to Smart Skin Analysis
            </Text>
            <Text variant="bodyLarge" style={styles.subtitle}>
              Our advanced AI-powered application helps identify potential skin conditions through
              sophisticated image analysis.
            </Text>
          </View>

          <Surface style={styles.featuresSection} elevation={1}>
            <Text variant="titleLarge" style={styles.sectionTitle}>
              Key Features
            </Text>
            
            <View style={styles.featureItem}>
              <MaterialCommunityIcons 
                name="robot" 
                size={32} 
                color={theme.colors.primary}
              />
              <View style={styles.featureText}>
                <Text variant="bodyLarge">AI-Powered Analysis</Text>
                <Text variant="bodyMedium" style={styles.featureDescription}>
                  Accurate skin condition assessment using advanced AI technology
                </Text>
              </View>
            </View>

            <View style={styles.featureItem}>
              <MaterialCommunityIcons 
                name="camera" 
                size={32} 
                color={theme.colors.primary}
              />
              <View style={styles.featureText}>
                <Text variant="bodyLarge">Easy Image Capture</Text>
                <Text variant="bodyMedium" style={styles.featureDescription}>
                  Simple and intuitive image upload functionality
                </Text>
              </View>
            </View>

            <View style={styles.featureItem}>
              <MaterialCommunityIcons 
                name="chart-box" 
                size={32} 
                color={theme.colors.primary}
              />
              <View style={styles.featureText}>
                <Text variant="bodyLarge">Detailed Reports</Text>
                <Text variant="bodyMedium" style={styles.featureDescription}>
                  Comprehensive analysis with personalized recommendations
                </Text>
              </View>
            </View>
          </Surface>

          <View style={styles.buttonContainer}>
            <Button
              mode="contained"
              onPress={() => navigation.navigate('Register')}
              style={styles.button}
              icon="account-plus"
            >
              Create Account
            </Button>
            <Button
              mode="outlined"
              onPress={() => navigation.navigate('Login')}
              style={styles.button}
              icon="login"
            >
              Sign In
            </Button>
          </View>
        </Surface>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    elevation: 4,
  },
  mainContent: {
    flex: 1,
    padding: 16,
  },
  contentContainer: {
    padding: 20,
    borderRadius: 15,
    marginBottom: 20,
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
  featuresSection: {
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
  },
  sectionTitle: {
    fontWeight: 'bold',
    marginBottom: 16,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  featureText: {
    marginLeft: 16,
    flex: 1,
  },
  featureDescription: {
    opacity: 0.7,
    marginTop: 4,
  },
  buttonContainer: {
    gap: 12,
    marginTop: 20,
  },
  button: {
    borderRadius: 8,
    paddingVertical: 6,
  },
});

export default MainScreen;
