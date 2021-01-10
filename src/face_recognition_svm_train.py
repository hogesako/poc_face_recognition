import face_recognition
from sklearn import svm
import pickle
import os
import gc
import time
import shutil
import PIL
from PIL import Image, ImageDraw

# Training the SVC classifier
encodings = []
names = []

# Training directory
work_dir_root = "/home/hogesako/poc_face_recognition/work/"
train_dir_root = work_dir_root + "train_dir/"
train_dir = os.listdir(train_dir_root)

start = time.time()
# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir(train_dir_root + person)

    # Loop through each training image for the current person
    for person_img in pix:
        img = Image.open(train_dir_root + person + "/" + person_img)
        if img.width > 1280:
            print("resized "+ str(img.width) + ":" + str(img.height))
            height = round(img.height * 1280 / img.width)
            img = img.resize((1280, height), resample=PIL.Image.BILINEAR)
            img.save(train_dir_root + person + "/" + person_img)
            del img
        img = Image.open(train_dir_root + person + "/" + person_img)
        if img.height > 1280:
            print("resized "+ str(img.width) + ":" + str(img.height))
            width = round(img.width * 1280 / img.height)
            img = img.resize((width, 1280), resample=PIL.Image.BILINEAR)
            img.save(train_dir_root + person + "/" + person_img)
        del img
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file(train_dir_root + person + "/" + person_img)

        print(person + "/" + person_img + " using cnn")
        face_bounding_boxes = face_recognition.face_locations(face, number_of_times_to_upsample=0, model="cnn")
        print(person + "/" + person_img + " finish cnn")
        if len(face_bounding_boxes) == 0:
            print(person + "/" + person_img + " using hog")
            face_bounding_boxes = face_recognition.face_locations(face, number_of_times_to_upsample=2)
            print(person + "/" + person_img + " finish hog")

        if len(face_bounding_boxes) == 0:
            print(person + "/" + person_img + " was skipped and can't be used for training. bounding_box length is " + str(len(face_bounding_boxes)))
            shutil.move(train_dir_root + person + "/" + person_img, work_dir_root + "no_faces/" + person)
        else:
            no = len(face_bounding_boxes)
            for i in range(no):
                face_enc = face_recognition.face_encodings(face, face_bounding_boxes, model="large")[i]
                encodings.append(face_enc)
                names.append(person)
            #shutil.move(train_dir_root + person + "/" + person_img, work_dir_root + "ok_faces/" + person)

        del face
        gc.collect()

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

with open(work_dir_root + 'model.pickle', mode='wb') as fp:
    pickle.dump(clf, fp)

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")