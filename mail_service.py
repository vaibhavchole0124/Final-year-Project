import smtplib
import ssl
import certifi
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

EMAIL_ADDRESS = "vaibhavchole0124"
EMAIL_PASSWORD = "sfibthccgwwqfand"

def send_otp_email(to_email, otp):
    subject = "Employee Attrition System - OTP"

    body = f"""
    <h2>Employee Attrition System</h2>
    <p>Your OTP is:</p>
    <h1 style="color:#d4a017;">{otp}</h1>
    <p>Valid for 5 minutes.</p>
    """

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html"))

    context = ssl.create_default_context(cafile=certifi.where())

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
