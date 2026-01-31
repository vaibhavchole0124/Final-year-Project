# test_email_debug.py
import os, smtplib
from email.message import EmailMessage

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT   = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER   = os.environ.get("SMTP_USER")
SMTP_PASS   = os.environ.get("SMTP_PASS")

msg = EmailMessage()
msg["Subject"] = "SMTP debug test"
msg["From"] = SMTP_USER
msg["To"] = SMTP_USER
msg.set_content("ही एक debug टेस्ट आहे.")

try:
    smtp = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
    smtp.set_debuglevel(1)   # <- इथे SMTP protocol conversation दिसेल
    smtp.ehlo()
    smtp.starttls()
    smtp.ehlo()
    smtp.login(SMTP_USER, SMTP_PASS)
    smtp.send_message(msg)
    print("Email पाठवला!")
except Exception as e:
    print("SMTP error:", e)
finally:
    try: smtp.quit()
    except: pass
