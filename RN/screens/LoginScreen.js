import React, { useState, useContext } from 'react';
import { View, StyleSheet, SafeAreaView } from 'react-native';
import { Text, TextInput, Button, useTheme, IconButton, Appbar } from 'react-native-paper';
import { ThemeContext } from '../context/ThemeContext';
import { Feather } from '@expo/vector-icons';
import { signInWithEmailAndPassword } from 'firebase/auth';
import auth from '../services/firebase';

const LoginScreen = ({ navigation }) => {
  const theme = useTheme();
  const { isDarkTheme, toggleTheme } = useContext(ThemeContext);
  const [form, setForm] = useState({ email: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const handleInputChange = (field, value) => {
    setForm({ ...form, [field]: value });
    setError('');
  };
  const handleLogin = async () => {
    if (!form.email || !form.password) {
      setError('Please fill in all fields');
      return;
    }
    setLoading(true);
    setError('');
    try {
      await signInWithEmailAndPassword(auth, form.email, form.password);
    } catch (error) {
      let errorMessage = 'Login failed. Please try again.';
      switch (error.code) {
        case 'auth/user-not-found':
        case 'auth/wrong-password':
          errorMessage = 'Invalid email or password';
          break;
        case 'auth/invalid-email':
          errorMessage = 'Please enter a valid email address';
          break;
        case 'auth/too-many-requests':
          errorMessage = 'Too many failed attempts. Please try again later';
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
        <Appbar.Content title="Login" />
        <IconButton icon={() => (<Feather name={isDarkTheme ? 'sun' : 'moon'} size={24} color={theme.colors.primary} />)} onPress={toggleTheme} />
      </Appbar.Header>
      <View style={styles.container}>
        <Text style={[styles.welcomeText, { color: theme.colors.primary }]}>Welcome Back</Text>
        <TextInput label="Email" value={form.email} onChangeText={(text) => handleInputChange('email', text)} keyboardType="email-address" mode="outlined" style={styles.input} autoCapitalize="none" />
        <TextInput label="Password" value={form.password} onChangeText={(text) => handleInputChange('password', text)} secureTextEntry mode="outlined" style={styles.input} />
        {error ? (<Text style={[styles.errorText, { color: theme.colors.error }]}>{error}</Text>) : null}
        <Button mode="contained" onPress={handleLogin} style={styles.button} loading={loading} disabled={loading}>
          Login
        </Button>
        <Button mode="text" onPress={() => navigation.navigate('Register')} style={styles.linkButton}>
          Don't have an account? Login here
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
export default LoginScreen;
