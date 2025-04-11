import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, Button } from 'react-native-paper';
import { MaterialCommunityIcons } from '@expo/vector-icons';

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

const styles = StyleSheet.create({
  errorContainer: { 
    flex: 1, 
    justifyContent: 'center', 
    alignItems: 'center', 
    padding: 20 
  },
  errorText: { 
    marginVertical: 16, 
    textAlign: 'center', 
    fontSize: 16 
  },
  errorButton: { 
    marginTop: 12 
  },
});

export default ErrorBoundary;