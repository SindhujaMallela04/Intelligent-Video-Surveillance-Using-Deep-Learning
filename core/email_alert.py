# email_alert.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "hudsinzoro@gmail.com"
SENDER_PASSWORD = "iwau ltkg uozg ggqa"  # Use an app password
RECIPIENT_EMAIL = "mallelasindhuja@gmail.com"

def send_email_alert(detected_action, confidence):
    subject = "⚠️ Alert: Suspicious Activity Detected!"
    body = f"Anomaly Detected: {detected_action}\nConfidence: {confidence:.2f}%\n\nCheck the surveillance footage immediately."

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print(f"✅ Email sent to {RECIPIENT_EMAIL} regarding {detected_action} alert.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
