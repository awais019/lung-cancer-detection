from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, UploadFile, Body, Header, HTTPException
from PIL import Image
import uvicorn
from pydantic import BaseModel
import sqlite3
import jwt
import datetime
from fastapi.responses import FileResponse
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model.load_state_dict(torch.load('resnet18_lung_cancer.pth'))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterRequest(BaseModel):
    name: str
    email: str
    age: int
    weight: float
    password: str


class SignInRequest(BaseModel):
    email: str
    password: str


def get_db_connection():
    return sqlite3.connect('lung_sense.db')


# Ensure users table exists
def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        age INTEGER NOT NULL,
        weight REAL NOT NULL,
        password TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()


init_db()

# Ensure reports table exists


def init_reports_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        prediction TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()


init_reports_db()

SECRET_KEY = "1234567890abcdef"
ALGORITHM = "HS256"

# Ensure reports directory exists
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Mount static files for reports (public access)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")


@app.get("/reports/{file_path:path}")
def serve_report(file_path: str):
    # This route is optional, but can be used to redirect or add custom logic
    return RedirectResponse(url=f"/reports/{file_path}")


@app.post("/register")
async def register(user: RegisterRequest):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        print(
            f"Inserting user: {user.name}, {user.email}, {user.age}, {user.weight}")
        c.execute('INSERT INTO users (name, email, age, weight, password) VALUES (?, ?, ?, ?, ?)',
                  (user.name, user.email, user.age, user.weight, user.password))
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        payload = {
            "user_id": user_id,
            "email": user.email,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        return {"success": True, "message": "User registered successfully", "token": token, "name": user.name}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Email already registered"}


@app.post("/signin")
async def signin(credentials: SignInRequest):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT id, name, password FROM users WHERE email = ?',
              (credentials.email,))
    row = c.fetchone()
    conn.close()
    if row and row[2] == credentials.password:
        user_id = row[0]
        user_name = row[1]
        payload = {
            "user_id": user_id,
            "email": credentials.email,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        return {"success": True, "message": "Signin successful", "token": token, "name": user_name}
    elif row:
        # Email exists but password is wrong
        raise HTTPException(status_code=401, detail="Invalid password", headers={
                            "WWW-Authenticate": "Bearer"})
    else:
        # Email does not exist
        raise HTTPException(status_code=404, detail="User not found")


# class 0: adenocarcinoma left lower lobe
# class 1: large cell carcinoma
# class 2: normal
# class 3: squamous cell carcinoma


@app.post("/predict")
async def predict(file: UploadFile, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Authorization token missing or invalid")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = payload.get("user_id")
    # Fetch user info
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT name, email, age, weight FROM users WHERE id = ?', (user_id,))
    user_row = c.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
    user_info = {
        "name": user_row[0],
        "email": user_row[1],
        "age": user_row[2],
        "weight": user_row[3]
    }
    # Predict
    image = Image.open(file.file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    class_map = {
        0: "adenocarcinoma left lower lobe",
        1: "large cell carcinoma",
        2: "normal",
        3: "squamous cell carcinoma"
    }
    risk_map = {
        "adenocarcinoma left lower lobe": {"risk": "High", "next_scan": "PET-CT in 1 month"},
        "large cell carcinoma": {"risk": "Moderate", "next_scan": "CT in 3 months"},
        "squamous cell carcinoma": {"risk": "High", "next_scan": "PET-CT in 1 month"},
        "normal": {"risk": "Low", "next_scan": "Routine CT in 12 months"}
    }
    prediction = class_map.get(predicted.item(), str(predicted.item()))
    risk_info = risk_map.get(
        prediction, {"risk": "Unknown", "next_scan": "Consult physician"})
    # Generate PDF
    import uuid
    filename = f"report_{user_id}_{uuid.uuid4().hex}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)
    pdf = canvas.Canvas(filepath, pagesize=letter)
    pdf.setTitle("Lung Cancer Prediction Report")
    pdf.drawString(50, 750, "Lung Cancer Prediction Report")
    pdf.drawString(50, 720, f"Name: {user_info['name']}")
    pdf.drawString(50, 700, f"Email: {user_info['email']}")
    pdf.drawString(50, 680, f"Age: {user_info['age']}")
    pdf.drawString(50, 660, f"Weight: {user_info['weight']}")
    pdf.drawString(50, 630, f"Prediction: {prediction}")
    pdf.drawString(50, 610, f"Risk Level: {risk_info['risk']}")
    pdf.drawString(50, 590, f"Next Recommended Scan: {risk_info['next_scan']}")
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    pdf.drawImage(ImageReader(img_buffer), 50, 400, width=200, height=200)
    pdf.showPage()
    pdf.save()
    # Save report record to DB
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('INSERT INTO reports (user_id, filename, prediction) VALUES (?, ?, ?)',
              (user_id, filename, prediction))
    conn.commit()
    # Fetch last scan date (previous report for this user, excluding current)
    c.execute('SELECT created_at FROM reports WHERE user_id = ? AND filename != ? ORDER BY created_at DESC LIMIT 1', (user_id, filename))
    last_scan_row = c.fetchone()
    conn.close()
    last_scan_date = last_scan_row[0] if last_scan_row else None
    url = f"/reports/{filename}"
    return {"success": True, "report_url": url, "prediction": prediction, "risk_level": risk_info['risk'], "next_scan": risk_info['next_scan'], "last_scan_date": last_scan_date}


@app.get("/history")
async def history(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Authorization token missing or invalid")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = payload.get("user_id")
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT filename, prediction, created_at FROM reports WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
    rows = c.fetchall()
    conn.close()
    history = [{
        "report_url": f"/reports/{row[0]}",
        "prediction": row[1],
        "created_at": row[2]
    } for row in rows]
    return {"success": True, "history": history}


@app.get("/")
async def root():
    return {"message": "Hello World"}


__main__ = "app"
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
