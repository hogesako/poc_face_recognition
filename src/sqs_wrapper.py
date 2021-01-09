# coding:utf-8
import boto3
import json


class SQSWrapper:

    def __init__(self):
        self.sqs = boto3.client('sqs', region_name='ap-northeast-1')
        self.rt_queue_url = self.sqs.get_queue_url(
            QueueName='retweet')['QueueUrl']
        self.img_queue_url = self.sqs.get_queue_url(
            QueueName='image_analysis.fifo')['QueueUrl']

    def retweet(self, tweet_id):
        self.sqs.send_message(
            QueueUrl=self.rt_queue_url,
            DelaySeconds=0,
            MessageBody=(tweet_id))

    def fetch_tweet(self):
        response = self.sqs.receive_message(
            QueueUrl=self.img_queue_url,
            AttributeNames=['SentTimestamp'],
            MaxNumberOfMessages=10,
            MessageAttributeNames=['All'],
            WaitTimeSeconds=20
        )

        list = []
        if ('Messages' in response) == False:
            return list

        messages = response['Messages']
        for message in messages:
            body = message["Body"]
            list.append(json.loads(body))

            receipt_handle = message['ReceiptHandle']
            self.sqs.delete_message(
                QueueUrl=self.img_queue_url,
                ReceiptHandle=receipt_handle
            )
        return list