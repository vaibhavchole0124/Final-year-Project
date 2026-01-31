import os, ssl, smtplib
from email.message import EmailMessage

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT   = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER   = os.environ.get("SMTP_USER")
SMTP_PASS   = os.environ.get("SMTP_PASS")
EMAIL_FROM  = os.environ.get("EMAIL_FROM", SMTP_USER)
TO = SMTP_USER  # send to same inbox for testing

print("Using SMTP_SERVER:", SMTP_SERVER, "SMTP_PORT:", SMTP_PORT)
print("Using SMTP_USER:", SMTP_USER)
print("SMTP_PASS length:", len(SMTP_PASS) if SMTP_PASS else "None")

if not SMTP_USER or not SMTP_PASS:
    print("ERROR: SMTP_USER or SMTP_PASS not set")
    raise SystemExit(1)

msg = EmailMessage()
msg["From"] = EMAIL_FROM
msg["To"] = TO
msg["Subject"] = "Test email from HR app"
msg.set_content("If you see this, SMTP worked!")

ctx = ssl.create_default_context()

try:
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
        server.set_debuglevel(1)
        server.ehlo()
        if SMTP_PORT == 587:
            server.starttls(context=ctx)
            server.ehlo()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
    print("Email sent OK")
except Exception as e:
    print("SMTP error:", repr(e))
