import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  StyleSheet,
  KeyboardAvoidingView,
  ScrollView,
  Platform,
  SafeAreaView,
  Dimensions,
} from 'react-native';
import {
  Text,
  TextInput,
  Button,
  RadioButton,
  useTheme,
  Appbar,
  Snackbar,
  Menu,
  TouchableRipple,
} from 'react-native-paper';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';
import { DatePickerInput } from 'react-native-paper-dates';

const Details = () => {
  const theme = useTheme();
  const navigation = useNavigation();
  const { height } = Dimensions.get('window');

  // Form fields state
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [dob, setDob] = useState('');
  const [gender, setGender] = useState('');
  const [address, setAddress] = useState('');
  const [phoneNo, setPhoneNo] = useState('');
  const [city, setCity] = useState('');
  const [medicalHistory, setMedicalHistory] = useState('');
  const [insured, setInsured] = useState('no'); // 'yes' or 'no'
  const [notes, setNotes] = useState('');

  // Gender dropdown menu state
  const [genderMenuVisible, setGenderMenuVisible] = useState(false);

  // Misc state
  const [error, setError] = useState('');
  const [snackbarVisible, setSnackbarVisible] = useState(false);
  const [contentHeight, setContentHeight] = useState(height * 1.5);
  const [currentStep, setCurrentStep] = useState(0); // 0-indexed step

  // Refs for inputs
  const lastNameRef = useRef(null);
  const addressRef = useRef(null);
  const phoneNoRef = useRef(null);
  const cityRef = useRef(null);
  const medicalHistoryRef = useRef(null);
  const notesRef = useRef(null);

  // Load user details on mount
  useEffect(() => {
    const loadDetails = async () => {
      try {
        const userDataString = await AsyncStorage.getItem('userData');
        if (userDataString) {
          const user = JSON.parse(userDataString);
          const res = await fetch(`http://192.168.215.143:5000/getDetails?email=${user.email}`);
          const data = await res.json();
          if (data.details) {
            const details = data.details;
            setFirstName(details.patient.firstName || '');
            setLastName(details.patient.lastName || '');
            setDob(details.patient.dob || '');
            setGender(details.patient.gender || '');
            setAddress(details.contact.address || '');
            setPhoneNo(details.contact.phone_no || '');
            setCity(details.contact.city || '');
            if (details.record) {
              setMedicalHistory(details.record.medical_history || '');
              setInsured(details.record.insured ? 'yes' : 'no');
              setNotes(details.record.notes || '');
            }
          }
        }
      } catch (e) {
        console.error(e);
      }
    };
    loadDetails();
  }, []);

  // Handle content size change (for scrolling)
  const onContentSizeChange = (contentWidth, newContentHeight) => {
    setContentHeight(Math.max(newContentHeight, height * 1.5));
  };

  // Validate required fields for a specific step
  const validateStep = () => {
    if (currentStep === 0) {
      if (!firstName.trim() || !lastName.trim() || !dob.trim()) {
        setError('Please enter First Name, Last Name, and Date of Birth.');
        return false;
      }
    } else if (currentStep === 1) {
      if (!address.trim() || !phoneNo.trim() || !city.trim()) {
        setError('Please enter Address, Phone Number, and City.');
        return false;
      }
    }
    // For step 3 there are no required fields
    setError('');
    return true;
  };

  // Navigation functions for multi-step form
  const goToNextStep = () => {
    if (!validateStep()) {
      return;
    }
    setCurrentStep((prev) => Math.min(prev + 1, steps.length - 1));
  };

  const goToPreviousStep = () => {
    setError('');
    setCurrentStep((prev) => Math.max(prev - 1, 0));
  };

  // Handle date change without timezone adjustment
  const handleDateChange = (date) => {
    if (!date) {
      setDob('');
      return;
    }
    
    // Directly format the local date to YYYY-MM-DD format.
    const formattedDate = date.toISOString().split('T')[0];
    setDob(formattedDate);
  };

  // Handle submit on final step
  const handleSubmit = async () => {
    if (!validateStep()) {
      return;
    }
    // Final check for all required fields
    if (!firstName.trim() || !lastName.trim() || !dob.trim() || !address.trim() || !phoneNo.trim() || !city.trim()) {
      setError('Please fill in all compulsory fields.');
      return;
    }
    try {
      const userDataString = await AsyncStorage.getItem('userData');
      if (!userDataString) {
        setError('User not found. Please login again.');
        return;
      }
      const user = JSON.parse(userDataString);
      const payload = {
        email: user.email,
        firstName,
        lastName,
        dob,
        gender,
        address,
        phone_no: phoneNo,
        city,
        medical_history: medicalHistory,
        insured: insured === 'yes',
        notes,
      };
      const res = await fetch('http://192.168.215.143:5000/updateUser', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await res.json();
      if (res.ok) {
        setSnackbarVisible(true);
      } else {
        setError(result.error || 'Failed to update details');
      }
    } catch (err) {
      setError('Failed to update details');
    }
  };

  // Gender selector component that uses Paper components for theming
  const GenderSelector = () => {
    return (
      <View style={styles.input}>
        <Text style={styles.label}>Gender</Text>
        <Menu
          visible={genderMenuVisible}
          onDismiss={() => setGenderMenuVisible(false)}
          anchor={
            <TouchableRipple
              onPress={() => setGenderMenuVisible(true)}
              style={[
                styles.genderSelector,
                { 
                  backgroundColor: theme.dark ? theme.colors.surface : theme.colors.background,
                  borderColor: theme.colors.outline,
                }
              ]}
            >
              <View style={styles.genderSelectorContent}>
                <Text style={{ color: theme.colors.onSurface }}>
                  {gender || 'Select Gender'}
                </Text>
              </View>
            </TouchableRipple>
          }
        >
          <Menu.Item onPress={() => { setGender('Male'); setGenderMenuVisible(false); }} title="Male" />
          <Menu.Item onPress={() => { setGender('Female'); setGenderMenuVisible(false); }} title="Female" />
          <Menu.Item onPress={() => { setGender('Other'); setGenderMenuVisible(false); }} title="Other" />
        </Menu>
      </View>
    );
  };

  // Steps of the form definition
  const steps = [
    {
      title: 'Step 1 of 3: Personal Info',
      content: (
        <>
          <TextInput
            label="First Name (Compulsory)"
            value={firstName}
            onChangeText={setFirstName}
            mode="outlined"
            style={styles.input}
            returnKeyType="next"
            onSubmitEditing={() => lastNameRef.current.focus()}
          />
          <TextInput
            ref={lastNameRef}
            label="Last Name (Compulsory)"
            value={lastName}
            onChangeText={setLastName}
            mode="outlined"
            style={styles.input}
            returnKeyType="next"
          />
          <View style={styles.dateContainer}>
            <Text style={styles.dateLabel}>Date of Birth (Compulsory)</Text>
            <DatePickerInput
              locale="en"
              value={dob ? new Date(dob) : undefined}
              onChange={handleDateChange}
              mode="outlined"
              inputMode="start"
              style={styles.dateInput}
            />
          </View>
        </>
      ),
    },
    {
      title: 'Step 2 of 3: Contact Details',
      content: (
        <>
          <TextInput
            ref={addressRef}
            label="Address (Compulsory)"
            value={address}
            onChangeText={setAddress}
            mode="outlined"
            style={styles.input}
            returnKeyType="next"
            onSubmitEditing={() => phoneNoRef.current.focus()}
          />
          <TextInput
            ref={phoneNoRef}
            label="Phone Number (Compulsory)"
            value={phoneNo}
            onChangeText={setPhoneNo}
            mode="outlined"
            style={styles.input}
            keyboardType="phone-pad"
            returnKeyType="next"
            onSubmitEditing={() => cityRef.current.focus()}
          />
          <TextInput
            ref={cityRef}
            label="City (Compulsory)"
            value={city}
            onChangeText={setCity}
            mode="outlined"
            style={styles.input}
            returnKeyType="done"
          />
        </>
      ),
    },
    {
      title: 'Step 3 of 3: Additional Details',
      content: (
        <>
          <GenderSelector />
          <TextInput
            ref={medicalHistoryRef}
            label="Medical History"
            value={medicalHistory}
            onChangeText={setMedicalHistory}
            mode="outlined"
            style={styles.input}
            multiline
            numberOfLines={3}
            returnKeyType="next"
            onSubmitEditing={() => notesRef.current.focus()}
          />
          <Text style={styles.label}>Insured</Text>
          <RadioButton.Group onValueChange={setInsured} value={insured}>
            <View style={styles.radioRow}>
              <RadioButton value="yes" />
              <Text style={styles.radioLabel}>Yes</Text>
              <RadioButton value="no" />
              <Text style={styles.radioLabel}>No</Text>
            </View>
          </RadioButton.Group>
          <TextInput
            ref={notesRef}
            label="Notes (e.g., Allergy details)"
            value={notes}
            onChangeText={setNotes}
            mode="outlined"
            style={styles.input}
            multiline
            numberOfLines={2}
            returnKeyType="done"
          />
        </>
      ),
    },
  ];

  return (
    <SafeAreaView style={[styles.mainContainer, { backgroundColor: theme.colors.background }]}>
      <Appbar.Header style={styles.header} mode="center-aligned">
        <Appbar.BackAction onPress={() => navigation.goBack()} />
        <Appbar.Content title="Update Details" />
      </Appbar.Header>
      <KeyboardAvoidingView
        style={styles.keyboardAvoidView}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 64 : 0}
      >
        <ScrollView
          contentContainerStyle={[styles.scrollContainer, { minHeight: contentHeight }]}
          onContentSizeChange={onContentSizeChange}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={true}
          persistentScrollbar={true}
        >
          <Text style={[styles.title, { color: theme.colors.primary }]}>
            {steps[currentStep].title}
          </Text>
          {steps[currentStep].content}
          {error ? (
            <Text style={[styles.errorText, { color: theme.colors.error }]}>{error}</Text>
          ) : null}
          <View style={styles.buttonRow}>
            {currentStep > 0 && (
              <Button mode="outlined" onPress={goToPreviousStep} style={styles.navButton}>
                Back
              </Button>
            )}
            {currentStep < steps.length - 1 ? (
              <Button mode="contained" onPress={goToNextStep} style={styles.navButton}>
                Next
              </Button>
            ) : (
              <Button mode="contained" onPress={handleSubmit} style={styles.navButton}>
                Submit
              </Button>
            )}
          </View>
          {/* Extra space at the bottom for better scrolling */}
          <View style={{ height: 80 }} />
        </ScrollView>
      </KeyboardAvoidingView>
      <Snackbar
        visible={snackbarVisible}
        onDismiss={() => {
          setSnackbarVisible(false);
          navigation.navigate('Home');
        }}
        duration={3000}
      >
        Details updated successfully!
      </Snackbar>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  mainContainer: {
    flex: 1,
    height: '100%',
  },
  header: {
    elevation: 4,
  },
  keyboardAvoidView: {
    flex: 1,
  },
  scrollContainer: {
    padding: 16,
    paddingBottom: 40,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  input: {
    marginBottom: 12,
    borderRadius: 4,
  },
  dateContainer: {
    marginBottom: 12,
  },
  dateLabel: {
    fontSize: 16,
    marginBottom: 4,
  },
  dateInput: {
    backgroundColor: 'transparent',
  },
  label: {
    marginBottom: 5,
    fontSize: 16,
  },
  radioRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  radioLabel: {
    marginRight: 20,
  },
  errorText: {
    marginVertical: 8,
    textAlign: 'center',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 16,
  },
  navButton: {
    flex: 1,
    marginHorizontal: 8,
  },
  genderSelector: {
    height: 56,
    borderWidth: 1,
    borderRadius: 4,
    justifyContent: 'center',
    paddingHorizontal: 14,
  },
  genderSelectorContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
});

export default Details;