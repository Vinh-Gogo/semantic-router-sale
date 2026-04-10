import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

class EmailRequest(BaseModel):
    to_email: str
    username: str = ""
    reset_link: str = ""

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/send-account-creation")
async def send_account_creation(req: EmailRequest):
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = req.to_email
        msg['Subject'] = "Tài khoản đã được tạo"
        
        body = f"Chào {req.username}, tài khoản của bạn đã sẵn sàng."
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return {"status": "success", "message": f"Đã gửi tới {req.to_email}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/send-forgot-password")
async def send_forgot_password(req: EmailRequest):
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = req.to_email
        msg['Subject'] = "Khôi phục mật khẩu"
        
        body = f"Link khôi phục: {req.reset_link}"
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}