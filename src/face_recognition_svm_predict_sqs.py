#!/usr/bin/env python3
import face_recognition
import os
import pickle
import time
import requests
import re
from PIL import Image
import io
import urllib.request
from sqs_wrapper import SQSWrapper

def analyze(tweet):
    tweet_id = str(tweet['id'])
    image_urls = tweet['image_urls']
    for image_url in image_urls:
        print(image_url)
        if img_analyze(image_url, tweet_id) == True:
            print('tweet id: ' + str(tweet_id))
            break

def img_analyze(image_url, tweet_id):
    try:
        response = urllib.request.urlopen(image_url)
    except urllib.error.URLError as e:
        return False
    test_image = face_recognition.load_image_file(response)
    face_locations = face_recognition.face_locations(test_image, number_of_times_to_upsample=0, model="cnn")
    no = len(face_locations)
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image, face_locations, model="large")[i]
        np_name = clf.predict([test_image_enc])
        name = np_name.tolist()[0]
        if name == 'kusudaaina':
            pil_image = Image.fromarray(test_image)
            split_path = os.path.split(image_url)
            pil_image.save(work_dir_root + 'save/' + tweet_id + "_" + split_path[1], quality=95)
            del pil_image
            return True
    return False

work_dir_root = "/home/hogesako/poc_face_recognition/work/"
with open(work_dir_root + 'model.pickle', mode='rb') as fp:
    clf = pickle.load(fp)

sqs = SQSWrapper()

while True:
    tweets = sqs.fetch_tweet()
    print('tweet count:' + str(len(tweets)))
    for tweet in tweets:
        analyze(tweet)
