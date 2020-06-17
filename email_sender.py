import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
import datetime
import os

sender_email = os.environ.get('EMAIL_ID')
print('Sending Email to: ', sender_email)
receiver_email = os.environ.get('EMAIL_ID')
password = os.environ.get('EMAIL_PWD')

message = MIMEMultipart("alternative")
message["Subject"] = "Alert: A New Person Entered the Premises"
message["From"] = sender_email
message["To"] = receiver_email


with open('pictures/demo.jpg', 'rb') as f:
    # set attachment mime and file name, the image type is png
    mime = MIMEBase('image', 'jpg', filename='img1.jpg')
    # add required header data:
    mime.add_header('Content-Disposition', 'attachment', filename='img1.jpg')
    mime.add_header('X-Attachment-Id', '0')
    mime.add_header('Content-ID', '<0>')
    # read attachment file content into the MIMEBase object
    mime.set_payload(f.read())
    # encode with base64
    encoders.encode_base64(mime)
    # add MIMEBase object to MIMEMultipart object
    message.attach(mime)


body = MIMEText('''
<html>
    <body>
        <h1>Alert</h1>
        <h2>A new has Person entered the Premises</h2>
        <h2>Body Temperature: 98.6</h2>
        <h2>Mask: Wearing</h2>
        <h2>Time: {}</h2>
        <p>
            <img src="cid:0">
        </p>
    </body>
</html>'''.format(datetime.datetime.now()), 'html', 'utf-8')

# Add HTML/plain-text parts to MIMEMultipart message
# The email client will try to render the last part first
message.attach(body)

# Create secure connection with server and send email
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(
        sender_email, receiver_email, message.as_string()
    )

