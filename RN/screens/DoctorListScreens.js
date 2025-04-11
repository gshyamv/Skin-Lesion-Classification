// screens/DoctorListScreen.js
import React, { useEffect, useState } from 'react';
import { View, StyleSheet, SafeAreaView, ScrollView } from 'react-native';
import { Text, Button, useTheme, Appbar, Divider } from 'react-native-paper';
import { useNavigation, useRoute } from '@react-navigation/native';

const DoctorListScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const route = useRoute();
  const { city } = route.params;  
  const [doctors, setDoctors] = useState([]);

  useEffect(() => {
    fetch(`http://192.168.215.143:5000/getDoctors?city=${encodeURIComponent(city)}`)
      .then(response => response.json())
      .then(data => {
        setDoctors(data.doctors);
      })
      .catch(error => {
        console.error('Error fetching doctors:', error);
      });
  }, [city]);

  const handleBookAppointment = (doctor) => {
    navigation.navigate('BookAppointment', { doctor: doctor });
  };

  return (
    <ScrollView style={{ backgroundColor: theme.colors.background }}>
      <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
        <Appbar.Header>
          <Appbar.BackAction onPress={() => navigation.goBack()} />
          <Appbar.Content title="Available Doctors" />
        </Appbar.Header>
        <ScrollView style={styles.container}>
          {doctors.map((doc, index) => (
            <View key={index} style={[styles.doctorCard, { backgroundColor: theme.colors.surface }]}>
              <Text style={styles.doctorName}>
                Dr. {doc.first_name} {doc.last_name}
              </Text>
              <Text>Clinic: {doc.clinic_name}</Text>
              <Text>City: {doc.city}</Text>
              <Text>Specialty: {doc.specialty}</Text>
              <Text>Experience: {doc.years_of_experience} years</Text>
              <Button
                mode="contained"
                onPress={() => handleBookAppointment(doc)}
                style={styles.bookButton}
              >
                Book an Appointment
              </Button>
              <Divider style={styles.divider} />
            </View>
          ))}
        </ScrollView>
      </SafeAreaView>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  container: {
    padding: 16,
  },
  doctorCard: {
    marginBottom: 16,
    padding: 16,
    borderRadius: 8,
    elevation: 2,
  },
  doctorName: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  bookButton: {
    marginTop: 8,
  },
  divider: {
    marginVertical: 8,
  },
});

export default DoctorListScreen;
