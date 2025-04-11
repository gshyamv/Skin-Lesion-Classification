import React, { useState, useEffect } from 'react';
import { View, StyleSheet, SafeAreaView, ScrollView, TouchableOpacity } from 'react-native';
import { Text, Button, useTheme, Appbar, Surface, Divider, Snackbar } from 'react-native-paper';
import { useNavigation, useRoute } from '@react-navigation/native';
import { format, addMonths, subMonths, startOfMonth, endOfMonth, eachDayOfInterval, isSameDay, isToday, isBefore } from 'date-fns';
import AsyncStorage from '@react-native-async-storage/async-storage';

const timeSlots = [
  '10:00 AM', '11:00 AM', '12:00 PM', '01:00 PM', 
  '02:00 PM', '03:00 PM', '04:00 PM', '05:00 PM'
];

// Simulated booked slots - in a real app, you'd fetch these from your backend
const getBookedSlots = (doctorId) => {
  return {
    '2025-04-12': ['10:00 AM', '02:00 PM'],
    '2025-04-13': ['11:00 AM', '03:00 PM'],
    '2025-04-14': ['01:00 PM', '04:00 PM'],
  };
};

const BookAppointmentScreen = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const route = useRoute();
  const { doctor } = route.params;

  const today = new Date();
  const [currentMonth, setCurrentMonth] = useState(today);
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedTime, setSelectedTime] = useState(null);
  const [viewMode, setViewMode] = useState('date'); // 'date' or 'time'
  const [bookedSlots, setBookedSlots] = useState({});
  const [patientEmail, setPatientEmail] = useState(null);
  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  useEffect(() => {
    // Fetch the user's email from AsyncStorage
    const fetchUserEmail = async () => {
      try {
        const userData = await AsyncStorage.getItem('userData');
        if (userData) {
          const { email } = JSON.parse(userData);
          setPatientEmail(email);
          console.log("Retrieved patient email:", email);
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
      }
    };

    fetchUserEmail();
    
    // In a real application, fetch the booked slots from your backend here
    setBookedSlots(getBookedSlots(doctor.doc_id));

    // Also fetch real booked slots if available
    fetchAvailableSlots();
  }, [doctor.doc_id]);

  const fetchAvailableSlots = async () => {
    if (selectedDate) {
      try {
        const formattedDate = format(selectedDate, 'yyyy-MM-dd');
        const response = await fetch(`http://192.168.215.143:5000/getAvailableSlots?doctor_id=${doctor.doc_id}&date=${formattedDate}`);
        if (response.ok) {
          const data = await response.json();
          const updatedBookedSlots = { ...bookedSlots };
          updatedBookedSlots[formattedDate] = data.booked_slots;
          setBookedSlots(updatedBookedSlots);
        }
      } catch (error) {
        console.error("Error fetching available slots:", error);
      }
    }
  };

  const onDateSelect = (day) => {
    setSelectedDate(day);
    setViewMode('time');
    const formattedDate = format(day, 'yyyy-MM-dd');
    fetch(`http://192.168.215.143:5000/getAvailableSlots?doctor_id=${doctor.doc_id}&date=${formattedDate}`)
      .then(response => response.json())
      .then(data => {
        const updatedBookedSlots = { ...bookedSlots };
        updatedBookedSlots[formattedDate] = data.booked_slots;
        setBookedSlots(updatedBookedSlots);
      })
      .catch(error => {
        console.error("Error fetching available slots:", error);
      });
  };

  const onTimeSelect = (time) => {
    setSelectedTime(time);
  };

  const handlePrevMonth = () => {
    setCurrentMonth(subMonths(currentMonth, 1));
  };

  const handleNextMonth = () => {
    setCurrentMonth(addMonths(currentMonth, 1));
  };

  const handleConfirmBooking = async () => {
    if (!selectedDate || !selectedTime) {
      setSnackbarMessage('Please select both date and time for your appointment.');
      setSnackbarVisible(true);
      return;
    }

    if (!patientEmail) {
      setSnackbarMessage('Your email could not be retrieved. Please log in again.');
      setSnackbarVisible(true);
      return;
    }

    try {
      const formattedDate = format(selectedDate, 'yyyy-MM-dd');
      console.log("Sending appointment request with:", {
        doctorId: doctor.doc_id,
        date: formattedDate,
        time: selectedTime,
        patientEmail: patientEmail
      });

      const response = await fetch('http://192.168.215.143:5000/bookAppointment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doctorId: doctor.doc_id,
          date: formattedDate,
          time: selectedTime,
          patientEmail: patientEmail
        })
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || 'Failed to book appointment');
      }

      setSnackbarMessage(`Your appointment with Dr. ${doctor.first_name} ${doctor.last_name} has been scheduled for ${format(selectedDate, 'MMMM d, yyyy')} at ${selectedTime}.`);
      setSnackbarVisible(true);
    } catch (error) {
      console.error('Error booking appointment:', error);
      setSnackbarMessage(error.message || 'There was an error booking your appointment. Please try again.');
      setSnackbarVisible(true);
    }
  };

  const renderCalendar = () => {
    const monthStart = startOfMonth(currentMonth);
    const monthEnd = endOfMonth(currentMonth);
    const dateRange = eachDayOfInterval({ start: monthStart, end: monthEnd });
    const weekDays = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
    const firstDayIndex = monthStart.getDay();
    const blanks = Array(firstDayIndex).fill(null);

    return (
      <View style={[styles.calendarContainer, { backgroundColor: theme.colors.surface }]}>
        <View style={styles.calendarHeader}>
          <TouchableOpacity onPress={handlePrevMonth}>
            <Text style={[styles.navigationArrow, { color: theme.colors.primary }]}>{'<'}</Text>
          </TouchableOpacity>
          <Text style={[styles.monthTitle, { color: theme.colors.text }]}>
            {format(currentMonth, 'MMMM, yyyy')}
          </Text>
          <TouchableOpacity onPress={handleNextMonth}>
            <Text style={[styles.navigationArrow, { color: theme.colors.primary }]}>{'>'}</Text>
          </TouchableOpacity>
        </View>
        
        <View style={styles.weekDaysContainer}>
          {weekDays.map((day, index) => (
            <Text key={index} style={[styles.weekDay, { color: theme.colors.text }]}>{day}</Text>
          ))}
        </View>
        
        <View style={styles.datesContainer}>
          {blanks.map((_, index) => (
            <View key={`blank-${index}`} style={styles.dateBox} />
          ))}
          
          {dateRange.map((date) => {
            const isSelected = selectedDate && isSameDay(date, selectedDate);
            const isDisabled = isBefore(date, today) && !isToday(date);
            const hasBookings = bookedSlots[format(date, 'yyyy-MM-dd')] && bookedSlots[format(date, 'yyyy-MM-dd')].length > 0;
            
            return (
              <TouchableOpacity
                key={date.toString()}
                style={[
                  styles.dateBox,
                  isSelected && [styles.selectedDateBox, { backgroundColor: theme.colors.primary }],
                  isDisabled && styles.disabledDateBox,
                  hasBookings && [styles.hasBookingsDateBox, { borderColor: theme.colors.error }]
                ]}
                onPress={() => !isDisabled && onDateSelect(date)}
                disabled={isDisabled}
              >
                <Text 
                  style={[
                    styles.dateText,
                    { color: theme.colors.text },
                    isToday(date) && [styles.todayText, { color: theme.colors.primary }],
                    isSelected && [styles.selectedDateText, { color: theme.colors.onPrimary }],
                    isDisabled && [styles.disabledDateText, { color: theme.colors.disabled }]
                  ]}
                >
                  {date.getDate()}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
        
        <View style={styles.legendContainer}>
          <TouchableOpacity 
            onPress={() => setViewMode('date')} 
            style={[styles.legendButton, { backgroundColor: theme.colors.surfaceVariant }]}
          >
            <Text style={[styles.legendText, { color: theme.colors.onSurfaceVariant }]}>Legend</Text>
          </TouchableOpacity>
          <View style={[styles.basketContainer, { backgroundColor: theme.colors.surfaceVariant }]}>
            <Text style={[styles.basketText, { color: theme.colors.onSurfaceVariant }]}>
              Basket ({selectedDate ? '1' : '0'} dates)
            </Text>
          </View>
        </View>
      </View>
    );
  };

  const renderTimeSelection = () => {
    const dateKey = selectedDate ? format(selectedDate, 'yyyy-MM-dd') : '';
    const bookedTimesForDate = bookedSlots[dateKey] || [];
    const scrollStyle = { maxHeight: 200 };

    return (
      <View style={[styles.timeSelectionContainer, { backgroundColor: theme.colors.surface }]}>
        <View style={styles.timeSelectionHeader}>
          <TouchableOpacity onPress={() => setViewMode('date')}>
            <Text style={[styles.navigationArrow, { color: theme.colors.primary }]}>{'<'}</Text>
          </TouchableOpacity>
          <Text style={[styles.timeSelectionTitle, { color: theme.colors.text }]}>
            {selectedDate ? format(selectedDate, 'do MMMM') : ''}
          </Text>
          <View style={{ width: 20 }} />
        </View>
        
        <ScrollView style={scrollStyle}>
          {timeSlots.map((time) => {
            const isSelected = selectedTime === time;
            const isBooked = bookedTimesForDate.includes(time);
            const isAvailable = !isBooked;
            return (
              <TouchableOpacity
                key={time}
                style={[
                  styles.timeSlot,
                  { backgroundColor: theme.colors.surfaceVariant },
                  isSelected && [styles.selectedTimeSlot, { backgroundColor: theme.colors.primaryContainer }],
                  isBooked && [styles.bookedTimeSlot, { backgroundColor: theme.colors.errorContainer }],
                  !isAvailable && styles.unavailableTimeSlot
                ]}
                onPress={() => isAvailable && onTimeSelect(time)}
                disabled={!isAvailable}
              >
                <Text style={{ color: theme.colors.onSurfaceVariant }}>{time}</Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>
        
        <View style={styles.legendContainer}>
          <View style={styles.timeLegendItem}>
            <View style={[styles.availableIndicator, { borderColor: theme.colors.primary }]} />
            <Text style={[styles.legendItemText, { color: theme.colors.text }]}>Available</Text>
          </View>
          <View style={styles.timeLegendItem}>
            <Text style={[styles.bookedX, { color: theme.colors.error }]}>✕</Text>
            <Text style={[styles.legendItemText, { color: theme.colors.text }]}>Booked</Text>
          </View>
          <View style={styles.timeLegendItem}>
            <View style={[styles.selectedIndicator, { backgroundColor: theme.colors.primary }]} />
            <Text style={[styles.legendItemText, { color: theme.colors.text }]}>Selected</Text>
          </View>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
      <Appbar.Header style={styles.header}>
        <Appbar.BackAction onPress={() => navigation.goBack()} />
        <Appbar.Content title="Book Appointment" />
      </Appbar.Header>
      
      <View style={styles.doctorInfoContainer}>
        <Text style={[styles.doctorName, { color: theme.colors.text }]}>
          Dr. {doctor.first_name} {doctor.last_name}
        </Text>
        <Text style={[styles.clinicInfo, { color: theme.colors.text }]}>
          {doctor.clinic_name} • {doctor.city}
        </Text>
        <Text style={[styles.specialtyInfo, { color: theme.colors.onSurfaceVariant }]}>
          {doctor.specialty} • {doctor.years_of_experience} years of experience
        </Text>
      </View>
      
      <Divider style={styles.divider} />
      
      <View style={styles.container}>
        {viewMode === 'date' ? renderCalendar() : renderTimeSelection()}
        
        <Button
          mode="contained"
          onPress={handleConfirmBooking}
          style={styles.confirmButton}
          disabled={!selectedDate || !selectedTime}
        >
          Confirm Booking
        </Button>
      </View>

      <Snackbar
        visible={snackbarVisible}
        onDismiss={() => {
          setSnackbarVisible(false);
          if (snackbarMessage.includes('has been scheduled')) {
            navigation.navigate('Home');
          }
        }}
        duration={3000}
        action={{
          label: 'OK',
          onPress: () => {
            setSnackbarVisible(false);
            if (snackbarMessage.includes('has been scheduled')) {
              navigation.navigate('Home');
            }
          },
        }}
      >
        {snackbarMessage}
      </Snackbar>
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
  container: {
    flex: 1,
    padding: 16,
  },
  doctorInfoContainer: {
    padding: 16,
  },
  doctorName: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  clinicInfo: {
    fontSize: 16,
    marginBottom: 2,
  },
  specialtyInfo: {
    fontSize: 14,
    opacity: 0.7,
  },
  divider: {
    marginVertical: 8,
  },
  calendarContainer: {
    borderRadius: 8,
    padding: 12,
    marginBottom: 20,
    elevation: 2,
  },
  calendarHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  navigationArrow: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  monthTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  weekDaysContainer: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  weekDay: {
    flex: 1,
    textAlign: 'center',
    fontWeight: 'bold',
  },
  datesContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  dateBox: {
    width: '14.28%',
    aspectRatio: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  dateText: {
    fontSize: 16,
  },
  selectedDateBox: {
    borderRadius: 8,
  },
  selectedDateText: {
    fontWeight: 'bold',
  },
  disabledDateBox: {
    opacity: 0.4,
  },
  disabledDateText: {
  },
  todayText: {
    fontWeight: 'bold',
  },
  hasBookingsDateBox: {
    borderWidth: 1,
    borderRadius: 8,
  },
  legendContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 16,
  },
  legendButton: {
    padding: 8,
    borderRadius: 4,
  },
  legendText: {
    fontWeight: 'bold',
  },
  basketContainer: {
    padding: 8,
    borderRadius: 4,
  },
  basketText: {
    fontWeight: 'bold',
  },
  timeSelectionContainer: {
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    elevation: 2,
  },
  timeSelectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  timeSelectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  timeSlot: {
    padding: 12,
    marginVertical: 4,
    borderRadius: 4,
    alignItems: 'center',
  },
  selectedTimeSlot: {
  },
  bookedTimeSlot: {
  },
  unavailableTimeSlot: {
    opacity: 0.5,
  },
  selectedIndicator: {
    width: 24,
    height: 24,
    borderRadius: 4,
  },
  bookedX: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  availableIndicator: {
    width: 24,
    height: 24,
    borderRadius: 4,
    borderWidth: 1,
  },
  timeLegendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 12,
  },
  legendItemText: {
    marginLeft: 4,
  },
  confirmButton: {
    marginTop: 16,
    paddingVertical: 6,
    borderRadius: 4,
  }
});

export default BookAppointmentScreen;