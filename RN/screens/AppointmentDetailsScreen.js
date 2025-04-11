// screens/AppointmentDetailsScreen.js

import React, { useEffect, useState } from 'react';
import { View, ScrollView, StyleSheet, SafeAreaView } from 'react-native';
import { Text, ActivityIndicator, Appbar, List, useTheme } from 'react-native-paper';
import { useNavigation } from '@react-navigation/native';
import { onAuthStateChanged } from 'firebase/auth';
import auth from '../services/firebase';

const AppointmentDetailsScreen = () => {
  const navigation = useNavigation();
  const theme = useTheme(); // Use the current theme
  const [appointments, setAppointments] = useState({ upcoming: [], past: [] });
  const [loading, setLoading] = useState(true);
  const [userEmail, setUserEmail] = useState('');

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user && user.email) {
        setUserEmail(user.email);
        fetchAppointments(user.email);
      } else {
        navigation.navigate('Login');
      }
    });
    return unsubscribe;
  }, [navigation]);

  const fetchAppointments = (email) => {
    fetch(`http://192.168.215.143:5000/getAppointments?email=${email}`)
      .then((res) => res.json())
      .then((data) => {
        setAppointments({
          upcoming: data.upcoming,
          past: data.past
        });
      })
      .catch((error) => {
        console.error("Error fetching appointments:", error);
      })
      .finally(() => setLoading(false));
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <Appbar.Header style={{ backgroundColor: theme.colors.primary }}>
        <Appbar.BackAction onPress={() => navigation.goBack()} />
        <Appbar.Content title="My Appointments" titleStyle={{ color: theme.colors.onPrimary }} />
      </Appbar.Header>
      {loading ? (
        <ActivityIndicator style={styles.loader} size="large" color={theme.colors.primary} />
      ) : (
        <ScrollView contentContainerStyle={styles.content}>
          <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Upcoming Appointments</Text>
          {appointments.upcoming.length === 0 ? (
            <Text style={[styles.emptyText, { color: theme.colors.text }]}>No upcoming appointments.</Text>
          ) : (
            appointments.upcoming.map((item) => (
              <List.Item
                key={item.appointment_id}
                title={`Dr. ${item.doctor.first_name} ${item.doctor.last_name}`}
                description={`${item.date} at ${item.time} - ${item.doctor.clinic_name}`}
                left={props => <List.Icon {...props} icon="calendar" />}
                titleStyle={{ color: theme.colors.text }}
                descriptionStyle={{ color: theme.colors.text }}
              />
            ))
          )}
          <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Past Appointments</Text>
          {appointments.past.length === 0 ? (
            <Text style={[styles.emptyText, { color: theme.colors.text }]}>No past appointments.</Text>
          ) : (
            appointments.past.map((item) => (
              <List.Item
                key={item.appointment_id}
                title={`Dr. ${item.doctor.first_name} ${item.doctor.last_name}`}
                description={`${item.date} at ${item.time} - ${item.doctor.clinic_name}`}
                left={props => <List.Icon {...props} icon="calendar-check" />}
                titleStyle={{ color: theme.colors.text }}
                descriptionStyle={{ color: theme.colors.text }}
              />
            ))
          )}
        </ScrollView>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  loader: { marginTop: 50 },
  content: { padding: 16 },
  sectionTitle: { fontSize: 18, fontWeight: 'bold', marginVertical: 12 },
  emptyText: { fontSize: 16, marginBottom: 12 }
});

export default AppointmentDetailsScreen;
