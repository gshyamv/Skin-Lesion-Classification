// screens/ResultScreen.js
import React, { useState } from 'react';
import { View, Image, StyleSheet, SafeAreaView } from 'react-native';
import { Appbar, Button, Text, ProgressBar, Portal, Dialog, Paragraph } from 'react-native-paper';
import { useNavigation, useRoute } from '@react-navigation/native';

const ResultScreen = () => {
  const navigation = useNavigation();
  const route = useRoute();
  // Expected route parameters: imageUri, cancerType, malignancyScore (a value from 0.0 to 1.0)
  const { imageUri, cancerType, malignancyScore } = route.params;

  const [dialogVisible, setDialogVisible] = useState(false);
  const hideDialog = () => setDialogVisible(false);

  return (
    <SafeAreaView style={styles.container}>
      <Appbar.Header style={styles.header}>
        <Appbar.Content title="Analysis Result" />
        <Appbar.Action 
          icon="robot"
          color="#6200ee" 
          onPress={() => setDialogVisible(true)} 
          accessibilityLabel="LLM Doctor"
        />
      </Appbar.Header>

      <View style={styles.contentContainer}>
        <Image source={{ uri: imageUri }} style={styles.image} resizeMode="cover" />
        <Text style={styles.cancerTypeText}>Diagnosis: {cancerType}</Text>
        <Text style={styles.progressLabel}>
          Malignancy: {(malignancyScore * 100).toFixed(0)}%
        </Text>
        <ProgressBar progress={malignancyScore} color="#B00020" style={styles.progressBar} />
        <Button mode="contained" style={styles.button} onPress={() => navigation.goBack()}>
          Back to Home
        </Button>
      </View>

      <Portal>
        <Dialog visible={dialogVisible} onDismiss={hideDialog}>
          <Dialog.Title>LLM Doctor</Dialog.Title>
          <Dialog.Content>
            <Paragraph>
              <Text style={{ fontWeight: 'bold' }}>Doctor Notes:</Text> The lesion appears to be classified as <Text style={{ fontStyle: 'italic' }}>{cancerType}</Text>. For further evaluation, we recommend consulting with a dermatologist.
            </Paragraph>
            <Paragraph style={styles.doctorListTitle}>Available Doctors:</Paragraph>
            <Paragraph>• Dr. Smith – Dermatology</Paragraph>
            <Paragraph>• Dr. Johnson – Oncology</Paragraph>
            <Paragraph>• Dr. Lee – Skin Specialist</Paragraph>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={hideDialog}>Close</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: { elevation: 4 },
  contentContainer: { flex: 1, padding: 16, alignItems: 'center' },
  image: { width: '100%', height: 300, borderRadius: 12, marginVertical: 16 },
  cancerTypeText: { fontSize: 20, fontWeight: 'bold', marginVertical: 8 },
  progressLabel: { fontSize: 16, marginVertical: 4 },
  progressBar: { width: '100%', height: 8, marginVertical: 12, borderRadius: 4 },
  button: { marginTop: 20, paddingHorizontal: 20 },
  doctorListTitle: { marginTop: 12, fontWeight: 'bold' },
});

export default ResultScreen;
