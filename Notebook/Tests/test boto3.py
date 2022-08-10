import boto3
from boto3.session import Session
import os

ACCESS_KEY = 'AKIAX6QQOVTSRS6MSP6L'
SECRET_KEY = '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'
REGION = 'eu-west-1'
BUCKET_NAME = 'md-ocr-p8-bucket'

session = Session(aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
s3 = session.resource('s3')
my_bucket = s3.Bucket(BUCKET_NAME)

for s3_file in my_bucket.objects.all():
    print(s3_file.key)


path = 's3://md-ocr-p8-bucket/Ressources'

print(os.listdir(path))