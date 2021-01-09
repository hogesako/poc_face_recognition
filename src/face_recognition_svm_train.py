import face_recognition
from sklearn import svm
import pickle
import os
import gc

# Training the SVC classifier
encodings = []
names = []

# Training directory
work_dir_root = "/root/work/"
train_dir_root = work_dir_root + "train_dir/"
train_dir = os.listdir(train_dir_root)

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir(train_dir_root + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file(train_dir_root + person + "/" + person_img)

        face_bounding_boxes = face_recognition.face_locations(face, number_of_times_to_upsample=2)
        if len(face_bounding_boxes) == 0:
            gc.collect()
            face_bounding_boxes = face_recognition.face_locations(face, number_of_times_to_upsample=0, model="cnn")

        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face, face_bounding_boxes)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")

        del face
        gc.collect()

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

with open(work_dir_root + 'model.pickle', mode='wb') as fp:
    pickle.dump(clf, fp)