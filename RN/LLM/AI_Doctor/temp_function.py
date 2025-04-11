import random

def get_random_diagnosis():
    diagnoses = {
        0: 'Actinic Keratosis (akiec)',
        1: 'Basal Cell Carcinoma (bcc)',
        2: 'Benign Keratosis (bkl)',
        3: 'Dermatofibroma (df)',
        4: 'Melanoma (mel)',
        5: 'Melanocytic Nevus (nv)',
        6: 'Vascular Lesion (vasc)'
    }
    random_key = random.choice(list(diagnoses.keys()))
    return diagnoses[random_key]