import os
import re
import random
from flask import Flask, render_template, request, redirect, url_for, jsonify
from google import genai
from dotenv import load_dotenv

from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)
from werkzeug.security import generate_password_hash, check_password_hash


# ---------------- CONFIG ----------------
load_dotenv()

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "supersecretkey123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///hospital.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ---------------- DATABASE MODELS ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100))
    doctor_name = db.Column(db.String(100))
    date = db.Column(db.String(20))
    time = db.Column(db.String(20))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------- GEMINI ----------------
gemini_client = None
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        gemini_client = genai.Client()
        print("Gemini initialized successfully")
except Exception as e:
    print("Gemini initialization failed:", e)


# ---------------- DOCTOR DATABASE ----------------
doctors = [
    {
        "name": "Dr. Anuja Sharma",
        "specialization": "Glioma",
        "experience": 15,
        "hospital": "Sterling Hospital"
    },
    {
        "name": "Dr. Ravi Kumar",
        "specialization": "Meningioma",
        "experience": 12,
        "hospital": "Apollo Hospital"
    },
    {
        "name": "Dr. Meera Reddy",
        "specialization": "Pituitary Tumor",
        "experience": 10,
        "hospital": "Manipal Hospital"
    },
    {
        "name": "Dr. Sanjay Patel",
        "specialization": "Glioma",
        "experience": 8,
        "hospital": "Fortis Hospital"
    },
    {
        "name": "Dr. Kavya Rao",
        "specialization": "Pituitary Tumor",
        "experience": 9,
        "hospital": "Aster Hospital"
    }
]


# ---------------- UTILITY FUNCTIONS ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    return None


def run_cnn_prediction(preprocessed_img):
    classes = ["Glioma", "Meningioma", "Pituitary Tumor", "No Tumor"]
    tumor = random.choice(classes)
    score = random.uniform(0.90, 0.99)
    return tumor, score


def get_risk_level(confidence):
    if confidence >= 0.97:
        return "High Risk"
    elif confidence >= 0.94:
        return "Moderate Risk"
    return "Low Risk"


def get_treatment(tumor_type):
    treatments = {
        "Glioma": "Surgery + Radiation Therapy",
        "Meningioma": "Observation or Surgery",
        "Pituitary Tumor": "Medication or Surgery",
        "No Tumor": "No treatment required"
    }
    return treatments.get(tumor_type, "Consult Specialist")


def get_treatment_cost(tumor_type):
    costs = {
        "Glioma": "₹3,50,000 - ₹7,00,000",
        "Meningioma": "₹2,00,000 - ₹5,00,000",
        "Pituitary Tumor": "₹2,50,000 - ₹6,00,000",
        "No Tumor": "₹0"
    }
    return costs.get(tumor_type, "Consult hospital")


def get_alert(risk):
    if risk == "High Risk":
        return "Immediate consultation recommended."
    elif risk == "Moderate Risk":
        return "Doctor consultation recommended within few days."
    return "No emergency detected."


def assign_doctors(tumor_type):
    matched = []

    for doc in doctors:
        score = random.randint(85, 99)
        doc_copy = doc.copy()
        doc_copy["score"] = score
        matched.append(doc_copy)

    matched.sort(key=lambda x: x["score"], reverse=True)
    return matched


def clean_report(text):
    if not text:
        return ""

    patterns = [
        r"PATIENT INFORMATION:.*?---",
        r"Electronically Signed By:.*",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    return text.strip()


def generate_medical_report(tumor_type, confidence):
    if gemini_client is None:
        return f"""
NEURORADIOLOGY REPORT

FINDINGS:
Detected tumor type: {tumor_type}

IMPRESSION:
Confidence score: {confidence*100:.2f}%

RECOMMENDATION:
Please consult neurologist for further diagnosis.
"""

    prompt = f"""
Generate neuroradiology report.

Tumor Type: {tumor_type}
Confidence: {confidence*100:.2f}%

Do not include patient details.
Do not include doctor signature.
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return clean_report(response.text)

    except Exception:
        return "Report generation failed."


# ---------------- AUTH ROUTES ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("index"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---------------- MAIN ROUTES ----------------
@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        preprocessed = preprocess_image(filepath)
        tumor, confidence = run_cnn_prediction(preprocessed)

        risk = get_risk_level(confidence)
        treatment = get_treatment(tumor)
        cost = get_treatment_cost(tumor)
        alert = get_alert(risk)
        doctor_list = assign_doctors(tumor)
        report = generate_medical_report(tumor, confidence)

        return render_template(
            "index.html",
            uploaded_image=url_for("static", filename=f"uploads/{filename}"),
            result_type=tumor,
            confidence=f"{confidence*100:.2f}%",
            risk=risk,
            treatment=treatment,
            treatment_cost=cost,
            alert_message=alert,
            doctors=doctor_list,
            full_report=report
        )

    return "Invalid file"


@app.route("/book", methods=["POST"])
@login_required
def book():
    patient_name = request.form.get("patient_name")
    doctor_name = request.form.get("doctor_name")
    date = request.form.get("date")
    time = request.form.get("time")

    appointment = Appointment(
        patient_name=patient_name,
        doctor_name=doctor_name,
        date=date,
        time=time
    )

    db.session.add(appointment)
    db.session.commit()

    return jsonify({
        "message": f"Appointment booked successfully with {doctor_name}"
    })


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get("message")

    return jsonify({
        "response": "Knowledge bot coming soon."
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        if User.query.filter_by(username="testuser").first() is None:
            user = User(username="testuser")
            user.set_password("password")
            db.session.add(user)
            db.session.commit()
            print("Created default user")

    app.run(debug=True)
