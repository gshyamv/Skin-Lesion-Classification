import React, { useState, useContext } from 'react';
import { View, StyleSheet, SafeAreaView } from 'react-native';
import { Text, TextInput, Button, useTheme, IconButton, Appbar } from 'react-native-paper';
import { ThemeContext } from '../context/ThemeContext';
import { Feather } from '@expo/vector-icons';
import { createUserWithEmailAndPassword, updateProfile } from 'firebase/auth';
import auth from '../services/firebase';

const RegisterScreen = ({ navigation }) => {
  const theme = useTheme();
  const { isDarkTheme, toggleTheme } = useContext(ThemeContext);
  const [form, setForm] = useState({ name: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const handleInputChange = (field, value) => {
    setForm({ ...form, [field]: value });
    setError('');
  };
  const validateForm = () => {
    if (!form.name || !form.email || !form.password) {
      setError('Please fill in all fields');
      return false;
    }
    if (form.password.length < 6) {
      setError('Password must be at least 6 characters long');
      return false;
    }
    if (!form.email.includes('@')) {
      setError('Please enter a valid email address');
      return false;
    }
    return true;
  };
  const handleRegister = async () => {
    if (!validateForm()) return;
    setLoading(true);
    setError('');
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, form.email, form.password);
      await updateProfile(userCredential.user, { displayName: form.name });
      await fetch('http://localhost:5000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: form.email, name: form.name }),
      });
    } catch (error) {
      let errorMessage = 'Registration failed. Please try again.';
      switch (error.code) {
        case 'auth/email-already-in-use':
          errorMessage = 'This email is already registered';
          break;
        case 'auth/invalid-email':
          errorMessage = 'Please enter a valid email address';
          break;
        case 'auth/weak-password':
          errorMessage = 'Password should be at least 6 characters';
          break;
        case 'auth/network-request-failed':
          errorMessage = 'Network error. Please check your connection';
          break;
      }
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };
  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
      <Appbar.Header style={styles.header} mode="center-aligned">
        <Appbar.BackAction onPress={() => navigation.goBack()} />
        <Appbar.Content title="Register" />
        <IconButton icon={() => (<Feather name={isDarkTheme ? 'sun' : 'moon'} size={24} color={theme.colors.primary} />)} onPress={toggleTheme} />
      </Appbar.Header>
      <View style={styles.container}>
        <Text style={[styles.welcomeText, { color: theme.colors.primary }]}>Create Your Account</Text>
        <TextInput label="Full Name" value={form.name} onChangeText={(text) => handleInputChange('name', text)} mode="outlined" style={styles.input} autoCapitalize="words" />
        <TextInput label="Email" value={form.email} onChangeText={(text) => handleInputChange('email', text)} keyboardType="email-address" mode="outlined" style={styles.input} autoCapitalize="none" />
        <TextInput label="Password" value={form.password} onChangeText={(text) => handleInputChange('password', text)} secureTextEntry mode="outlined" style={styles.input} />
        {error ? (<Text style={[styles.errorText, { color: theme.colors.error }]}>{error}</Text>) : null}
        <Button mode="contained" onPress={handleRegister} style={styles.button} loading={loading} disabled={loading}>
          Register
        </Button>
        <Button mode="text" onPress={() => navigation.navigate('Login')} style={styles.linkButton}>
          Already have an account? Login here
        </Button>
      </View>
    </SafeAreaView>
  );
};
const styles = StyleSheet.create({
  safeArea: { flex: 1 },
  header: { elevation: 4 },
  container: { flex: 1, padding: 20, justifyContent: 'center' },
  welcomeText: { fontSize: 28, fontWeight: 'bold', marginBottom: 30, textAlign: 'center' },
  input: { marginBottom: 15 },
  button: { marginTop: 10, paddingVertical: 6 },
  linkButton: { marginTop: 20 },
  errorText: { textAlign: 'center', marginBottom: 10 },
});
export default RegisterScreen;
