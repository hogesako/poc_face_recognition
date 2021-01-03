import face_recognition
import os
import pickle
import gc

from PIL import Image, ImageDraw

with open('model.pickle', mode='rb') as fp:
    clf = pickle.load(fp)

test_dir_root = "test_dir/"
test_dirs = os.listdir(test_dir_root)

for person_img in test_dirs:
    test_image = face_recognition.load_image_file(test_dir_root + person_img)

    pil_image = Image.fromarray(test_image)
    draw = ImageDraw.Draw(pil_image)

    print(person_img)
    face_locations = face_recognition.face_locations(test_image, number_of_times_to_upsample=2)
    if len(face_locations) == 0:
        face_locations = face_recognition.face_locations(test_image, number_of_times_to_upsample=0, model="cnn")
    print(person_img)
    gc.collect()
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        np_name = clf.predict([face_encoding])
        name = np_name.tolist()[0]
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    pil_image.save("test_out/"+ "outed_" + person_img)
    del draw
    del pil_image
    gc.collect()
