import face_recognition
import os
import pickle

with open('model.pickle', mode='rb') as fp:
    clf = pickle.load(fp)

# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('test_image.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    print(*name)
