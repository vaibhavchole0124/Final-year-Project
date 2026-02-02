import os
import io
import zipfile
import sqlite3
import psycopg2
import psycopg2.extras
from urllib.parse import urlparse
import smtplib
import random
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
from dotenv import load_dotenv
load_dotenv()

# EMAIL_SENDER and PASSWORD handled via os.environ below

from functools import wraps

from collections import defaultdict

# reportlab higher-level imports for nicer PDF tables
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm


from flask import (
    Flask, render_template, request, redirect, session, url_for, send_file, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
import numpy as np
import joblib

# PDF support
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# SHAP support
import shap
SHAP_EXPLAINER = None

# ---------------------- EMAIL + OTP CONFIG -----------------------
import ssl
import time
from email.message import EmailMessage
import os

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT   = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER   = os.environ.get("EMAIL_SENDER", "") # Use EMAIL_SENDER as SMTP_USER
SMTP_PASS   = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_FROM  = os.environ.get("EMAIL_FROM", SMTP_USER)

OTP_LENGTH = 6
OTP_TTL_SECONDS = 5 * 60        # 5 minutes
OTP_MAX_RESENDS = 3
# ----------------------------------------------------------------

import os, sqlite3



def get_db():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        # PostgreSQL Connection
        conn = psycopg2.connect(db_url)
        return conn
    else:
        # SQLite Fallback (Local Dev)
        db_path = os.path.join(os.path.dirname(__file__), "users.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

class CursorWrapper:
    def __init__(self, cursor, is_postgres):
        self.cursor = cursor
        self.is_postgres = is_postgres

    def execute(self, query, params=None):
        if self.is_postgres:
            query = query.replace("?", "%s")
        if params is None:
            return self.cursor.execute(query)
        return self.cursor.execute(query, params)

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()

    def __getattr__(self, name):
        return getattr(self.cursor, name)

def get_cursor(conn):
    # Helper to get a cursor that works for both (DictCursor for Postgres to mimic sqlite3.Row)
    is_postgres = not isinstance(conn, sqlite3.Connection)
    if is_postgres:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    else:
        cur = conn.cursor()
    return CursorWrapper(cur, is_postgres)


def save_prediction(username, result, probability, prediction_type="Single"):
    conn = get_db()
    cur = get_cursor(conn)

    cur.execute("""
        INSERT INTO prediction_logs (username, prediction_type, result, probability)
        VALUES (?, ?, ?, ?)
    """, (username, prediction_type, result, probability))

    conn.commit()
    conn.close()


def generate_otp(length=OTP_LENGTH):
    start = 10 ** (length - 1)
    end = (10 ** length) - 1
    return str(random.randint(start, end))


ctx = ssl.create_default_context(cafile=certifi.where())

def send_email_otp(to_email, subject, body):
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM or SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    ctx = ssl.create_default_context()

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            if SMTP_PORT == 587:
                server.starttls(context=ctx)
                server.ehlo()
            if SMTP_USER and SMTP_PASS:
                server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)


# -----------------------------------------------------------------------------
#  BASIC CONFIGURATION
# -----------------------------------------------------------------------------
app = Flask(__name__)
# Secret key for session. Replace with a secure random string for production.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_dev_secret")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor_rf.pkl")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}

LAST_BULK_DF = None






# -----------------------------------------------------------------------------
#  FEATURE DEFINITIONS
# -----------------------------------------------------------------------------
FEATURE_COLS = [
    "Age", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField",
    "EnvironmentSatisfaction",
    "Gender", "HourlyRate", "JobInvolvement", "JobLevel",
    "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "OverTime",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
    "MaternityPaternityLeave", "WorkplaceHarassment", "RemoteWorkFrequency",
    "MentalHealthResources", "ProjectDeadlinePressure", "SkillDevelopmentHours"
]

NUMERIC_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "MaternityPaternityLeave", "WorkplaceHarassment", "RemoteWorkFrequency",
    "MentalHealthResources", "ProjectDeadlinePressure", "SkillDevelopmentHours"
]

CATEGORICAL_FEATURES = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus",
    "Over18", "OverTime"
]


# -----------------------------------------------------------------------------
#  LOAD MODEL + PREPROCESSOR
# -----------------------------------------------------------------------------
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    print("✅ Model and Preprocessor loaded successfully!")
    
    # Init SHAP
    try:
        # Use TreeExplainer for Random Forest
        # We need to pass the model, but TreeExplainer handles sklearn models directly usually.
        # But wait, our model is a pipeline? No, it's just the model object in the pickle.
        # Let's verify what 'model' is. If it's RandomForestClassifier, this works.
        SHAP_EXPLAINER = shap.TreeExplainer(model)
        print("✅ SHAP Explainer initialized!")
    except Exception as e:
        print(f"⚠️ SHAP Init Failed: {e}")
except Exception as e:
    print(f"⚠️ Warning: Could not load model/preprocessor — {e}")
    preprocessor = None
    model = None


# -----------------------------------------------------------------------------
#  SHAP / PREPROCESSOR COLUMN HELPERS
# -----------------------------------------------------------------------------
def get_preprocessor_output_columns(preprocessor_obj):
    """
    Try to extract final column names after preprocessing
    (numeric + one-hot categorical). Fallback to FEATURE_COLS.
    """
    try:
        # Adjust keys "num"/"cat" to whatever you used in ColumnTransformer
        num_cols = []
        cat_cols = []

        if "num" in preprocessor_obj.named_transformers_:
            num_cols = list(
                preprocessor_obj.named_transformers_["num"].get_feature_names_out()
            )
        if "cat" in preprocessor_obj.named_transformers_:
            cat_cols = list(
                preprocessor_obj.named_transformers_["cat"].get_feature_names_out()
            )

        cols = num_cols + cat_cols
        if len(cols) == 0:
            raise ValueError("No columns from preprocessor.")
        return cols
    except Exception as e:
        print("⚠️ Preprocessor column extraction error:", e)
        # Fallback – lengths may not match exactly but avoids crash
        return FEATURE_COLS


if preprocessor is not None:
    PREP_COLUMNS = get_preprocessor_output_columns(preprocessor)
else:
    PREP_COLUMNS = FEATURE_COLS


def map_shap_indices_to_names(values_vec):
    values_vec = np.array(values_vec).ravel()
    abs_vals = np.abs(values_vec)

    if abs_vals.size == 0:
        return [], []

    TOP_K = 5
    idxs = np.argsort(abs_vals)[-TOP_K:][::-1]

    labels = []
    vals = []

    for idx in idxs:
        if idx < len(PREP_COLUMNS):
            labels.append(PREP_COLUMNS[int(idx)])
        else:
            labels.append(f"Feature {int(idx) + 1}")
        # absolute SHAP impact
        vals.append(float(abs(values_vec[int(idx)])))

    return labels, vals






# -----------------------------------------------------------------------------
#  UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_single_dataframe(form_data):
    """Convert HTML form → DataFrame (not used in index now but kept)."""
    row = {}

    for col in FEATURE_COLS:
        val = form_data.get(col)
        if col in NUMERIC_FEATURES:
            try:
                row[col] = float(val)
            except Exception:
                row[col] = np.nan
        else:
            row[col] = val

    return pd.DataFrame([row])


def run_model(df):
    """Run preprocessor + model. Returns probability, label, risk score."""
    if preprocessor is None or model is None:
        raise RuntimeError("Model not loaded!")

    X = preprocessor.transform(df)
    probs = model.predict_proba(X)[:, 1]  # probability of LEAVE
    labels = np.where(probs >= 0.5, "Leave", "Stay")
    risk_scores = (probs * 100).round(1)
    return probs, labels, risk_scores


def get_risk_bucket(score):
    """Map score to risk level."""
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

# ---------- PDF helper (use reportlab platypus Table) ----------
def df_to_pdf_bytes(df: pd.DataFrame, title: str = "Report"):
    """
    Convert a pandas DataFrame into a multi-page PDF (bytes) using ReportLab platypus Table.
    Returns BytesIO ready to be sent by Flask.
    """
    buffer = io.BytesIO()
    # landscape A4 gives more horizontal space
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=1*cm, rightMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    max_col_chars = 20
    max_cell_chars = 50

    cols = [c if len(c) <= max_col_chars else c[:max_col_chars-3] + "..." for c in df.columns.tolist()]

    data = [cols]
    for _, row in df.iterrows():
        row_vals = []
        for c in df.columns:
            v = row[c]
            s = "" if pd.isna(v) else str(v)
            s = " ".join(s.split())
            if len(s) > max_cell_chars:
                s = s[:max_cell_chars-3] + "..."
            row_vals.append(s)
        data.append(row_vals)

    page_width = landscape(A4)[0] - doc.leftMargin - doc.rightMargin
    ncols = max(1, len(cols))
    col_width = max(3*cm, page_width / ncols)
    col_widths = [col_width] * ncols

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table_style = TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ])
    table.setStyle(table_style)

    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer


# -----------------------------------------------------------------------------
#  RECOMMENDATION ENGINE (ADVANCED)
# -----------------------------------------------------------------------------
def make_recommendations(row, score):
    recs = []

    # Main risk-based suggestion
    if score >= 80:
        recs.append("Critical risk – Immediate HR intervention required.")
    elif score >= 60:
        recs.append("High risk – Review workload and engagement plan.")
    elif score >= 40:
        recs.append("Medium risk – Monitor performance and morale.")
    else:
        recs.append("Low risk – Continue regular engagement.")

    # Helper functions
    def f(col, default=0):
        try:
            return float(row.get(col, default))
        except Exception:
            return default

    def s(col, default=""):
        try:
            return str(row.get(col, default)).strip()
        except Exception:
            return default

    # Job satisfaction
    if f("JobSatisfaction") <= 2:
        recs.append("Low job satisfaction – Improve recognition & communication.")

    # Work-life balance
    if f("WorkLifeBalance") <= 2:
        recs.append("Poor work-life balance – Offer flexibility or WFH option.")

    # Overtime
    if s("OverTime") == "Yes":
        recs.append("Excessive overtime – Reduce workload and redistribute tasks.")

    # Promotion gap
    if f("YearsSinceLastPromotion") >= 4:
        recs.append("Long time without promotion – Consider career development plan.")

    # Early career
    if f("TotalWorkingYears") <= 3:
        recs.append("Early career employee – Provide mentoring and skill development.")

    # New Features
    if f("WorkplaceHarassment") == 1:
        recs.append("⚠️ Uses reported Workplace Harassment – Immediate investigation required.")

    if f("RemoteWorkFrequency") == 0:
        recs.append("Zero remote work – Consider offering hybrid options.")
    
    if f("MaternityPaternityLeave") == 1:
        recs.append("Returning from parental leave – Ensure smooth reintegration support.")

    # Round 2 Features
    if f("MentalHealthResources") == 0:
        recs.append("Lack of mental health support – Consider introducing wellness programs.")
    
    if f("ProjectDeadlinePressure") >= 3:
        recs.append("High deadline pressure – Review workload distribution to prevent burnout.")
    
    if f("SkillDevelopmentHours") < 5:
        recs.append("Low skill development – Increases stagnation risk. Encourage training.")

    # Salary-related
    if f("MonthlyIncome") < 3000:
        recs.append("Low salary – Review compensation adjustments.")

    return recs
 # ============================= Email-OTP ======================


    



 # ============================= REGISTER / LOGIN / LOGOUT ======================
import re
from werkzeug.security import generate_password_hash


@app.route("/register", methods=["GET", "POST"])
def register():
    """
    Register a new user
    IMPORTANT: Email is used as Username
    """
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        # ---------- BASIC VALIDATION ----------
        if not email or not password or not confirm:
            return render_template(
                "register.html",
                error="All fields are required."
            )

        if password != confirm:
            return render_template(
                "register.html",
                error="Password and Confirm Password do not match."
            )

        # ---------- PASSWORD STRENGTH ----------
        if len(password) < 8:
            return render_template(
                "register.html",
                error="Password must be at least 8 characters long."
            )

        if not re.search(r"[A-Z]", password):
            return render_template(
                "register.html",
                error="Password must contain at least one uppercase letter (A-Z)."
            )

        if not re.search(r"[0-9]", password):
            return render_template(
                "register.html",
                error="Password must contain at least one number (0-9)."
            )

        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return render_template(
                "register.html",
                error="Password must contain at least one special character (@, #, $, etc.)."
            )

        # ---------- DATABASE ----------
        conn = get_db()
        cur = get_cursor(conn)


        # Ensure table exists
        # Note: SQLite uses AUTOINCREMENT, Postgres uses SERIAL (or GENERATED BY DEFAULT AS IDENTITY)
        # We need dialect-specific DDL or a generic one. 
        # For this quick migration, let's just check table existence safely.
        
        # We'll rely on a manual init or check. 
        # But to keep the auto-init behavior:
        if isinstance(conn, sqlite3.Connection):
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT,
                    email TEXT
                )
            """)
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE,
                    password TEXT,
                    email TEXT
                )
            """)
        conn.commit()

        # Email = Username
        username = email

        # Check if user already exists
        cur.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            conn.close()
            return render_template(
                "register.html",
                error="This email is already registered."
            )

        # Save user with hashed password
        hashed = generate_password_hash(password)
        cur.execute(
            "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
            (username, hashed, email)
        )
        conn.commit()
        conn.close()

        return render_template(
            "register.html",
            info="Registration successful! Please login."
        )

    return render_template("register.html")



# ---------- LOGIN (single, DB-backed) ----------


# ---------------- LOGIN (STEP 1: SEND OTP) ----------------
# REPLACE your existing @app.route("/login", methods=["GET","POST"]) function with this block (dev/debug helper)

@app.route("/login", methods=["GET", "POST"])
def login():
    # quick dev / safety: if already logged in, redirect
    if session.get("logged_in"):
        return redirect(url_for("index"))

    error = None
    try:
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            if not username or not password:
                error = "Enter username and password."
                return render_template("login.html", error=error)

            # DB check
            conn = get_db()
            cur = get_cursor(conn)
            cur.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cur.fetchone()
            conn.close()

            # validate credentials
            if not user:
                error = "Invalid username or password."
                return render_template("login.html", error=error)

            # user may be sqlite3.Row — use mapping safely
            try:
                user_map = dict(user)
            except Exception:
                # fallback: try indexing, then wrap minimal dict
                try:
                    user_map = {"username": user[1], "password": user[2]}
                except Exception:
                    user_map = {}

            stored_hash = user_map.get("password") or user.get("password") if hasattr(user, "__getitem__") else None

            if not stored_hash or not check_password_hash(stored_hash, password):
                error = "Invalid username or password."
                return render_template("login.html", error=error)

            # Credentials OK → Generate OTP
            otp = generate_otp()
            now = int(time.time())

            session["otp_data"] = {
                "username": user_map.get("username") or user[1],
                "user_id": user_map.get("id") if "id" in user_map else user[0],
                "role": user_map.get("role") if "role" in user_map else user[4],
                "otp": otp,
                "created_at": now,
                "resend_count": 0
            }


            # pick recipient email from DB (email column must exist) safely
            recipient_email = user_map.get("email") if isinstance(user_map, dict) else None

            if not recipient_email:
                # fallback: if username looks like email use it
                if "@" in username and "." in username.split("@")[-1]:
                    recipient_email = username
                else:
                    # For debugging - show message instead of failing silently
                    return render_template(
                        "login.html",
                        error="No email associated with this account. Please register with an email or contact admin."
                    )

            subject = "Your Login OTP"
            body = f"Your OTP is: {otp}\nValid for 5 minutes."

            # DEV mode: show OTP on screen instead of sending email if env var set
            if os.environ.get("DEV_MODE_SHOW_OTP", "") == "1":
                return render_template("verify_otp.html", info=f"DEV MODE: OTP is {otp}")

            sent, err = send_email_otp(recipient_email, subject, body)
            if not sent:
                # show the returned smtp error text in page for debugging
                return render_template("login.html", error=f"Failed to send OTP: {err}")

            return render_template("verify_otp.html", info=f"OTP sent to {recipient_email}")

    except Exception as exc:
        # catch-any — show message AND print to server console for full traceback
        import traceback
        tb = traceback.format_exc()
        print("--- LOGIN EXCEPTION TRACEBACK ---")
        print(tb)
        # expose a short message to browser (dev only)
        return render_template("login.html", error=f"Internal error: {str(exc)}. See server logs for details.")

    return render_template("login.html", error=error)


@app.route("/home")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("home.html")

# ---------------- STEP 2: VERIFY OTP ----------------
@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    otp_input = request.form.get("otp", "").strip()
    otp_state = session.get("otp_data")

    if not otp_state:
        return render_template("login.html", error="OTP session expired. Login again.")

    now = int(time.time())
    if now - otp_state["created_at"] > OTP_TTL_SECONDS:
        session.pop("otp_data", None)
        return render_template("login.html", error="OTP expired. Login again.")

    if otp_input == otp_state["otp"]:
        session["logged_in"] = True
        session["username"] = otp_state["username"]
        session["role"] = otp_state["role"]   # ✅ ADD THIS

        session.pop("otp_data", None)

    # ✅ redirect based on role
    if session["role"] == "admin":
        return redirect(url_for("admin_dashboard"))
    else:
        return redirect(url_for("home"))


@app.route("/resend-otp", methods=["POST"])
def resend_otp():
    try:
        otp_data = session.get("otp_data")

        if not otp_data:
            return render_template("verify_otp.html", error="Session expired. Please login again.")

        # Avoid unlimited resends
        resend_count = otp_data.get("resend_count", 0)
        if resend_count >= 3:
            return render_template("verify_otp.html", error="Resend limit reached. Please login again.")

        # Generate new OTP
        new_otp = generate_otp()
        otp_data["otp"] = new_otp
        otp_data["created_at"] = int(time.time())
        otp_data["resend_count"] = resend_count + 1
        session["otp_data"] = otp_data

        username = otp_data.get("username")

        # Fetch email address from DB
        conn = get_db()
        cur = get_cursor(conn)
        cur.execute("SELECT email FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()

        if not user or not user["email"]:
            return render_template("verify_otp.html", error="No email found for this account.")

        recipient_email = user["email"]

        # Prepare message
        subject = "Your OTP (Resent)"
        body = f"Your new OTP is: {new_otp}\nValid for 5 minutes."

        # ====== DEBUG MODE (DEV_MODE_SHOW_OTP=1) ============
        if os.environ.get("DEV_MODE_SHOW_OTP") == "1":
            return render_template("verify_otp.html", info=f"DEV MODE: OTP is {new_otp}")

        # Send email
        sent, err = send_email_otp(recipient_email, subject, body)

        if not sent:
            return render_template("verify_otp.html", error=f"Failed to resend OTP. ({err})")

        return render_template("verify_otp.html", info=f"OTP resent to {recipient_email}")

    except Exception as e:
        print("RESEND OTP ERROR:", e)
        return render_template("verify_otp.html", error="Unexpected error while resending OTP.")

# ============================= SINGLE PREDICTION (HOME) ======================


# Default values for fields not in the form (based on dataset modes/medians)
DEFAULT_VALUES = {
    "Age": 30,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 802,
    "Department": "Research & Development",
    "DistanceFromHome": 9,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 66,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 3,
    "MaritalStatus": "Single",
    "MonthlyIncome": 6500,
    "MonthlyRate": 14313,
    "NumCompaniesWorked": 3,
    "Over18": "Y",
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 3,
    "MaternityPaternityLeave": 0,
    "WorkplaceHarassment": 0,
    "RemoteWorkFrequency": 0,
    "MentalHealthResources": 0,
    "ProjectDeadlinePressure": 2,
    "SkillDevelopmentHours": 5
}

@app.route("/", methods=["GET", "POST"])
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    prediction = None

    if request.method == "POST":
        try:
            row = {}
            for col in FEATURE_COLS:
                # Get from form, OR use default, OR None
                val_from_form = request.form.get(col)
                if val_from_form is None or val_from_form.strip() == "":
                    # Try default
                    value = DEFAULT_VALUES.get(col)
                else:
                    value = val_from_form

                if col in NUMERIC_FEATURES:
                    try:
                        row[col] = float(value)
                    except Exception:
                        # logical fallback if default is also somehow missing/bad
                        row[col] = 0.0 
                else:
                    row[col] = value

            df = pd.DataFrame([row])

            if preprocessor is None or model is None:
                raise RuntimeError("Model or preprocessor not loaded.")

            X = preprocessor.transform(df)
            prob = float(model.predict_proba(X)[0][1])
            risk_score = round(prob * 100, 2)

            # 🔹 RESULT
            if prob >= 0.5:
                label = "Likely to LEAVE"
                badge_class = "danger"
                chart_color = "#dc3545"
                prediction_result = "At Risk"
            else:
                label = "Likely to STAY"
                badge_class = "success"
                chart_color = "#198754"
                prediction_result = "Safe"

            # Explain with SHAP
            explanation = []
            if SHAP_EXPLAINER and prob >= 0.5:
                # Transform just this row
                X_shap = preprocessor.transform(df)
                # Calculate SHAP values
                shap_values = SHAP_EXPLAINER.shap_values(X_shap)
                
                # shap_values is list for classification [val_class0, val_class1]
                # We want class 1 (Leave)
                # Handle binary case:
                if isinstance(shap_values, list):
                    vals = shap_values[1][0] # Class 1, first (only) row
                else:
                    # If older/newer shap version returns matrix directly
                    vals = shap_values[0] if len(shap_values.shape)==2 else shap_values[1][0] 

                # Map column names
                feature_names = get_preprocessor_output_columns(preprocessor)
                
                # Sort absolute imp
                # But we care about POSITIVE contribution to LEAVE (vals > 0)
                # Let's take top 3 contributors
                # import pandas as pd  <-- REMOVED: Caused UnboundLocalError
                shap_df = pd.DataFrame(list(zip(feature_names, vals)), columns=['Feature', 'SHAP'])
                # Filter positive (driving attrition)
                shap_df = shap_df[shap_df['SHAP'] > 0].sort_values(by='SHAP', ascending=False).head(3)
                
                for _, r in shap_df.iterrows():
                    feat_clean = r['Feature'].split('__')[-1] # cleaner name
                    explanation.append(feat_clean)


            # ✅ SAVE PREDICTION LOG
            save_prediction(
                session.get("username"),
                prediction_result,
                risk_score,
                "Single"
            )

            prediction = {
                "label": label,
                "probability": f"{prob:.3f}",
                "risk_score": risk_score,
                "badge_class": badge_class,
                "chart_color": chart_color,
                "recommendations": make_recommendations(df.iloc[0], risk_score),
                "key_drivers": explanation # Add this
            }

        except Exception as e:
            prediction = {"error": f"Prediction failed: {e}"}

    return render_template("index.html", prediction=prediction)


# ============================= BULK PREDICTION ===============================
@app.route("/bulk", methods=["GET", "POST"])
def bulk():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    error = None
    summary = None
    chart = None
    table = None
    columns = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("bulk.html", error="Please choose a file to upload.")

        if not allowed_file(file.filename):
            return render_template("bulk.html", error="Unsupported file type. Upload CSV/Excel only.")

        # Read file
        try:
            ext = file.filename.rsplit(".", 1)[1].lower()
            df = pd.read_csv(file) if ext == "csv" else pd.read_excel(file)
        except Exception as e:
            return render_template("bulk.html", error=f"Error reading file: {e}")

        # Check columns
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            return render_template("bulk.html", error="Missing columns: " + ", ".join(missing))

        # Enforce EmployeeNumber for identification
        if "EmployeeNumber" not in df.columns:
            return render_template("bulk.html", error="Missing required identification column: EmployeeNumber")

        # Run model
        df_features = df[FEATURE_COLS].copy()
        try:
            probs, labels, risk_scores = run_model(df_features)
        except Exception as e:
            return render_template("bulk.html", error=f"Model prediction error: {e}")

        # Build result DF
        df_result = df.copy()
        df_result["AttritionPrediction"] = np.where(labels == "Leave", "Likely to LEAVE", "Likely to STAY")
        df_result["AttritionProbability"] = np.round(probs, 3)
        df_result["RiskScore"] = risk_scores

        # Buckets + Recommendations
        buckets = []
        recs_list = []
        for (_, row), score in zip(df_features.iterrows(), risk_scores):
            buckets.append(get_risk_bucket(float(score)))
            recs = make_recommendations(row, float(score))
            recs_list.append("; ".join(recs))

        df_result["RiskBucket"] = buckets
        df_result["Recommendations"] = recs_list

        # Reorder columns: Put EmployeeNumber first, then Predictions/Risk, then Features
        cols = ["EmployeeNumber", "AttritionPrediction", "RiskScore", "RiskBucket", "Recommendations"] + \
               [c for c in df.columns if c not in ["EmployeeNumber", "AttritionPrediction", "RiskScore", "RiskBucket", "Recommendations"]]
        
        df_result = df_result[cols]

        # Ensure output folder
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save main bulk output
        df_result.to_csv(os.path.join(OUTPUT_DIR, "bulk_predictions.csv"), index=False)
        df_result.to_excel(os.path.join(OUTPUT_DIR, "bulk_predictions.xlsx"), index=False, engine="openpyxl")

        # ---- SAVE leave.csv / stay.csv / summary.csv (DOWNLOAD BUTTONS USE THESE) ----
        leave_df = df_result[df_result["AttritionPrediction"] == "Likely to LEAVE"]
        stay_df = df_result[df_result["AttritionPrediction"] == "Likely to STAY"]

        leave_df.to_csv(os.path.join(OUTPUT_DIR, "leave.csv"), index=False)
        stay_df.to_csv(os.path.join(OUTPUT_DIR, "stay.csv"), index=False)

        # Summary calculation
        stay_count = (labels == "Stay").sum()
        leave_count = (labels == "Leave").sum()
        total = len(df_result)

        stay_pct = round(stay_count * 100 / total, 1)
        leave_pct = round(leave_count * 100 / total, 1)

        risk_count = df_result["RiskBucket"].value_counts()
        low_cnt = risk_count.get("Low", 0)
        med_cnt = risk_count.get("Medium", 0)
        high_cnt = risk_count.get("High", 0)
        crit_cnt = risk_count.get("Critical", 0)

        # Save summary.csv
        summary_df = pd.DataFrame([{
            "TotalEmployees": total,
            "Stay": stay_count,
            "Leave": leave_count,
            "StayPct": stay_pct,
            "LeavePct": leave_pct,
            "LowRisk": low_cnt,
            "MediumRisk": med_cnt,
            "HighRisk": high_cnt,
            "CriticalRisk": crit_cnt,
        }])

        summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "bulk_summary.csv"), index=False)

        # Update global for employee report routes
        global LAST_BULK_DF
        LAST_BULK_DF = df_result.copy()

        # UI summary
        summary = {
            "total": total,
            "staying": stay_count,
            "leaving": leave_count,
            "stay_pct": stay_pct,
            "leave_pct": leave_pct,
            "risk_low": low_cnt,
            "risk_med": med_cnt,
            "risk_high": high_cnt,
            "risk_crit": crit_cnt,
        }

        chart = {"stay": stay_count, "leave": leave_count}

        # Only show first 100 leave cases
        leave_only = leave_df.head(100)
        table = leave_only.to_dict(orient="records")
        columns = list(leave_only.columns)

        # 🔔 PROACTIVE ALERT: Send email if Critical Risk found
        if crit_cnt > 0:
             try:
                 critical_ids = df_result[df_result["RiskBucket"] == "Critical"]["EmployeeNumber"].tolist()
                 # Limit to first 20 for email brevity
                 ids_str = ", ".join(map(str, critical_ids[:20]))
                 if len(critical_ids) > 20: 
                     ids_str += f" ... and {len(critical_ids)-20} more."

                 # Get user email
                 conn_alert = get_db()
                 cur_alert = get_cursor(conn_alert)
                 cur_alert.execute("SELECT email FROM users WHERE username = ?", (session.get("username"),))
                 u_alert = cur_alert.fetchone()
                 conn_alert.close()
                 
                 if u_alert and u_alert["email"]:
                     subject = f"⚠️ CRITICAL ALERT: {crit_cnt} Employees at High Risk"
                     body = f"""
                     ATTN: HR Admin,

                     The recent bulk analysis detected {crit_cnt} employees with CRITICAL attrition risk.
                     
                     Employee IDs:
                     {ids_str}

                     Please review the full report immediately on the dashboard.

                     - Employee Attrition System
                     """
                     if os.environ.get("DEV_MODE_SHOW_OTP") != "1":
                         send_email_otp(u_alert["email"], subject, body)
                         print(f"📧 Alert sent to {u_alert['email']}")
             except Exception as e:
                 print(f"⚠️ Failed to send alert email: {e}")

    return render_template("bulk.html", error=error, summary=summary, chart=chart, table=table, columns=columns)


# ============================= HISTORY / ANALYTICS ===========================
@app.route("/history")
def history():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    
    conn = get_db()
    cur = get_cursor(conn)
    
    # Fetch user's history
    username = session.get("username")
    cur.execute("SELECT * FROM prediction_logs WHERE username = ? ORDER BY timestamp DESC LIMIT 50", (username,))
    rows = cur.fetchall()
    conn.close()
    
    return render_template("history.html", history=rows)




# ============================= DASHBOARD =====================================

@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    # ============================= SIMULATOR (WHAT-IF) ===========================
    pass

@app.route("/simulation")
def simulation():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("simulator.html")

@app.route("/api/predict_simulation", methods=["POST"])
def predict_simulation():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.json
        
        # Merge input with defaults to ensure all features exist
        # We start with DEFAULT_VALUES copy, then update with user input
        row = DEFAULT_VALUES.copy()
        for k, v in data.items():
            if k in row:
                row[k] = v
        
        # Convert to DF
        # Ensure Numeric types
        for col in NUMERIC_FEATURES:
            if col in row:
                try:
                    row[col] = float(row[col])
                except:
                    row[col] = 0.0

        # --- SMART LOGIC FOR SIMULATION ---
        # Ensure consistency so defaults don't mask risk
        
        # 1. If Age is young, TotalWorkingYears cannot be high
        if row.get("Age", 30) < 25:
            row["TotalWorkingYears"] = min(row.get("TotalWorkingYears", 0), 2)
            row["JobLevel"] = 1 # Juniors are usually level 1
            
        # 2. YearsAtCompany constraints
        tenure = row.get("YearsAtCompany", 5)
        # YearsInCurrentRole can't exceed Tenure
        if row.get("YearsInCurrentRole", 0) > tenure:
            row["YearsInCurrentRole"] = tenure
        # YearsWithCurrManager can't exceed Tenure
        if row.get("YearsWithCurrManager", 0) > tenure:
            row["YearsWithCurrManager"] = tenure
        # TotalWorkingYears must be >= Tenure
        if row.get("TotalWorkingYears", 0) < tenure:
            row["TotalWorkingYears"] = tenure
            
        # 3. If JobSatisfaction is low, likely JobInvolvement drops too (heuristic)
        if row.get("JobSatisfaction", 3) == 1:
            row["JobInvolvement"] = min(row.get("JobInvolvement", 3), 2)

        df = pd.DataFrame([row])
        
        # Predict
        if preprocessor is None or model is None:
            return jsonify({"error": "Model not loaded"}), 500

        X = preprocessor.transform(df)
        prob = float(model.predict_proba(X)[0][1])
        
        # DEBUG LOGGING
        print("------------- SIMULATION DEBUG -------------")
        print(f"Input Features: {row}")
        print(f"Predicted Probability: {prob}")
        print("--------------------------------------------")
        
        risk_score = round(prob * 100, 2)
        
        label = "Likely to LEAVE" if prob >= 0.5 else "Likely to STAY"
        
        return jsonify({
            "risk_score": risk_score,
            "probability": prob,
            "label": label
        })

    except Exception as e:
        print(f"Simulation Error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================= DASHBOARD =====================================

    # ... rest of your dashboard code unchanged from earlier ...

    """Analytics dashboard view"""

    global LAST_BULK_DF

    # --------- CHOOSE DATA SOURCE ----------
    # If bulk upload was done, use that data for dashboard
    if LAST_BULK_DF is not None:
        df = LAST_BULK_DF.copy()
        dataset_name = "Latest Bulk Uploaded Dataset"
    else:
        # fallback to default IBM dataset
        dataset_name = "IBM HR Attrition"

        # ---------- Load Dataset ----------
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception as e:
            return render_template(
                "dashboard.html",
                dataset_name=dataset_name,
                error=f"Dataset loading error: {e}",
                total_records=0,
                models_compared=0,
                best_accuracy=0,
                best_model_name=None,
                overall_labels=[],
                overall_values=[],
                dept_labels=[],
                dept_values=[],
                job_labels=[],
                job_values=[],
                tenure_labels=[],
                tenure_values=[],
                gender_labels=[],
                gender_values=[],
                overtime_labels=[],
                overtime_values=[],
                risk_bins_labels=[],
                risk_bins_values=[],
                risk_seg_labels=[],
                risk_seg_values=[],
                fi_labels=[],
                fi_values=[],
                salary_leave_values=[],
                salary_stay_values=[],
                dept_health_labels=[],
                dept_health_values=[],
                model_rows=[],
                model_names=[],
                model_acc_values=[],
                model_f1_values=[],
                leave_pct=0,
                stay_pct=0,
                forecast_labels=[],
                forecast_values=[],
                insights=[],        # <<< important
            )

    total_records = len(df)

    # ---------- Overall Attrition ----------
    stay = int((df["Attrition"] == "No").sum())
    leave = int((df["Attrition"] == "Yes").sum())
    total_emp = stay + leave if (stay + leave) > 0 else 1

    leave_pct = round(leave * 100 / total_emp, 1)
    stay_pct = round(stay * 100 / total_emp, 1)

    overall_labels = ["Stay", "Leave"]
    overall_values = [stay, leave]

    # ---------- Attrition by Department ----------
    dept_series = df[df["Attrition"] == "Yes"]["Department"].value_counts()
    dept_labels = dept_series.index.tolist()
    dept_values = dept_series.values.astype(int).tolist()

    # ---------- Attrition by Job Role ----------
    job_series = (
        df[df["Attrition"] == "Yes"]["JobRole"]
        .value_counts()
        .sort_values()
    )
    job_labels = job_series.index.tolist()
    job_values = job_series.values.astype(int).tolist()

    # ---------- Attrition rate by Tenure ----------
    tenure_series = (
        df.groupby("YearsAtCompany")["Attrition"]
        .apply(lambda s: (s == "Yes").mean() * 100)
    )
    tenure_labels = [int(x) for x in tenure_series.index.tolist()]
    tenure_values = [round(float(v), 2) for v in tenure_series.values.tolist()]

    # ---------- Attrition by Gender ----------
    gender_series = df.groupby("Gender")["Attrition"].apply(
        lambda s: (s == "Yes").mean() * 100
    )
    gender_labels = gender_series.index.tolist()
    gender_values = [round(float(v), 2) for v in gender_series.values.tolist()]

    # ---------- Attrition by OverTime ----------
    overtime_series = df.groupby("OverTime")["Attrition"].apply(
        lambda s: (s == "Yes").mean() * 100
    )
    overtime_labels = overtime_series.index.tolist()
    overtime_values = [round(float(v), 2) for v in overtime_series.values.tolist()]

    # ---------- Risk Distribution (using model) ----------
    risk_bins_labels = []
    risk_bins_values = []
    risk_seg_labels = []
    risk_seg_values = []

    if preprocessor is not None and model is not None:
        try:
            features = df[FEATURE_COLS]
            _, _, risk_scores = run_model(features)

            # Bins: 0–20–40–60–80–100
            bins = [0, 20, 40, 60, 80, 100]
            names = ["0–20", "20–40", "40–60", "60–80", "80–100"]

            counts, _ = np.histogram(risk_scores, bins=bins)
            risk_bins_labels = names
            risk_bins_values = counts.astype(int).tolist()

            # Segmentation
            buckets = [get_risk_bucket(float(x)) for x in risk_scores]
            seg = pd.Series(buckets).value_counts()

            risk_seg_labels = ["Low", "Medium", "High", "Critical"]
            risk_seg_values = [
                int(seg.get("Low", 0)),
                int(seg.get("Medium", 0)),
                int(seg.get("High", 0)),
                int(seg.get("Critical", 0)),
            ]
        except Exception as e:
            print("Risk Score Error:", e)

    # ---------- Feature Importance (Correlation-based) ----------
    try:
        target = (df["Attrition"] == "Yes").astype(int)
        corr = df[NUMERIC_FEATURES].corrwith(target).abs().sort_values(ascending=False)
        fi_labels = corr.index.tolist()[:10]
        fi_values = (corr.values[:10] * 100).round(2).tolist()
    except Exception:
        fi_labels, fi_values = [], []

    # ---------- Salary vs Attrition ----------
    salary_leave = df[df["Attrition"] == "Yes"]["MonthlyIncome"].tolist()
    salary_stay = df[df["Attrition"] == "No"]["MonthlyIncome"].tolist()

    # ---------- Department Health Score ----------
    try:
        dept_total = df.groupby("Department")["Attrition"].count()
        dept_leave = df[df["Attrition"] == "Yes"].groupby("Department")["Attrition"].count()

        leave_rate = (dept_leave / dept_total).fillna(0) * 100
        dept_health = 100 - (leave_rate * 2)

        dept_health_labels = dept_health.index.tolist()
        dept_health_values = dept_health.values.round(1).tolist()
    except Exception:
        dept_health_labels, dept_health_values = [], []

    # ---------- MODEL COMPARISON ----------
    model_rows = []
    model_names = []
    model_acc_values = []
    model_f1_values = []
    models_compared = 0
    best_model_name = None
    best_accuracy = 0

    comparison_path = os.path.join(BASE_DIR, "models", "model_comparison_results.csv")

    if os.path.exists(comparison_path):
        try:
            cmp_df = pd.read_csv(comparison_path)
            models_compared = len(cmp_df)

            cmp_df_display = cmp_df.copy()
            for col in ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "ROC_AUC"]:
                cmp_df_display[col] = (cmp_df_display[col] * 100).round(2)

            model_rows = cmp_df_display.to_dict(orient="records")

            best_idx = cmp_df["Accuracy"].idxmax()
            best_model_name = str(cmp_df.loc[best_idx, "Model"])
            best_accuracy = round(float(cmp_df.loc[best_idx, "Accuracy"] * 100), 2)

            model_names = cmp_df_display["Model"].tolist()
            model_acc_values = cmp_df_display["Accuracy"].tolist()
            model_f1_values = cmp_df_display["F1_macro"].tolist()
        except Exception:
            models_compared = 0
            best_model_name = None
            best_accuracy = 0

    # ---------- FORECAST ----------
    try:
        hist = df.groupby("YearsAtCompany")["Attrition"].apply(
            lambda s: (s == "Yes").mean() * 100
        )
        hist_values = [float(v) for v in hist.values.tolist()]
        hist_labels = [f"Year {int(v)}" for v in hist.index.tolist()]

        if len(hist_values) >= 3:
            avg = sum(hist_values[-3:]) / 3
        else:
            avg = sum(hist_values) / len(hist_values)

        forecast_values = hist_values + [avg] * 6
        forecast_labels = hist_labels + [f"Next {i}" for i in range(1, 7)]
    except Exception:
        forecast_labels = []
        forecast_values = []

    # ---------- INSIGHTS (TEXT) ----------
    insights = []
    try:
        insights.append(
            f"Overall attrition is {leave_pct}% leaving vs {stay_pct}% staying."
        )

        if not dept_series.empty:
            top_dept = dept_series.idxmax()
            top_dept_count = int(dept_series.max())
            insights.append(
                f"The highest number of exits is from the {top_dept} department ({top_dept_count} employees leaving)."
            )

        if not job_series.empty:
            top_role = job_series.idxmax()
            insights.append(
                f"The job role most affected by attrition is {top_role}."
            )

        # Overtime vs no overtime
        if len(overtime_values) == 2:
            rate_map = dict(zip(overtime_labels, overtime_values))
            ot_yes = rate_map.get("Yes")
            ot_no = rate_map.get("No")
            if ot_yes is not None and ot_no is not None:
                insights.append(
                    f"Employees working overtime have an attrition rate of about {ot_yes:.1f}% "
                    f"compared to {ot_no:.1f}% for employees without overtime."
                )

        if "WorkLifeBalance" in fi_labels:
            insights.append(
                "Work-life balance is one of the strongest numeric drivers of attrition."
            )
        if "JobSatisfaction" in fi_labels:
            insights.append(
                "Job satisfaction is highly correlated with attrition – focus on engagement initiatives."
            )

    except Exception as e:
        print("Insight generation error:", e)
        insights = []

    # ---------- INSIGHTS (KPI + DISTRIBUTIONS) ----------

    # 1. KPI Stability Factors
    kpi_low_satisfaction = int((df["JobSatisfaction"] <= 2).sum())
    kpi_poor_wlb = int((df["WorkLifeBalance"] <= 2).sum())
    kpi_overtime = int((df["OverTime"] == "Yes").sum())
    kpi_no_promo = int((df["YearsSinceLastPromotion"] >= 5).sum())
    kpi_low_salary = int((df["MonthlyIncome"] < 3000).sum())

    # KPI chart data
    stability_labels = [
        "Low Job Satisfaction",
        "Poor Work-Life Balance",
        "Overtime Employees",
        "No Promotion (5+ years)",
        "Low Salary Employees",
    ]
    stability_values = [
        kpi_low_satisfaction,
        kpi_poor_wlb,
        kpi_overtime,
        kpi_no_promo,
        kpi_low_salary,
    ]

    # 2. Age Distribution
    age_bins = ["20–30", "30–40", "40–50", "50+"]
    age_counts = [
        int(df[(df["Age"] >= 20) & (df["Age"] < 30)].shape[0]),
        int(df[(df["Age"] >= 30) & (df["Age"] < 40)].shape[0]),
        int(df[(df["Age"] >= 40) & (df["Age"] < 50)].shape[0]),
        int(df[df["Age"] >= 50].shape[0]),
    ]

    # 3. Income Distribution
    income_bins = ["0–3000", "3000–5000", "5000–8000", "8000+"]
    income_counts = [
        int(df[df["MonthlyIncome"] < 3000].shape[0]),
        int(df[(df["MonthlyIncome"] >= 3000) & (df["MonthlyIncome"] < 5000)].shape[0]),
        int(df[(df["MonthlyIncome"] >= 5000) & (df["MonthlyIncome"] < 8000)].shape[0]),
        int(df[df["MonthlyIncome"] >= 8000].shape[0]),
    ]

    # 4. Education Field Distribution
    edu_series = df["EducationField"].value_counts()
    edu_labels = edu_series.index.tolist()
    edu_values = edu_series.values.tolist()

    # 5. Job Level Distribution
    joblevel_series = df["JobLevel"].value_counts().sort_index()
    joblevel_labels = joblevel_series.index.tolist()
    joblevel_values = joblevel_series.values.tolist()

    # 6. Department Attrition Ranking
    dept_rank = df.groupby("Department")["Attrition"].apply(
        lambda s: (s == "Yes").mean() * 100
    )
    dept_names = dept_rank.index.tolist()
    dept_attrition = dept_rank.values.round(2).tolist()

    # ---------- RENDER ----------
    return render_template(
        "dashboard.html",

        dataset_name=dataset_name,
        total_records=total_records,
        leave_pct=leave_pct,
        stay_pct=stay_pct,

        # Overall
        overall_labels=overall_labels,
        overall_values=overall_values,

        # Department & Job
        dept_labels=dept_labels,
        dept_values=dept_values,
        job_labels=job_labels,
        job_values=job_values,

        # Tenure / Gender / Overtime
        tenure_labels=tenure_labels,
        tenure_values=tenure_values,
        gender_labels=gender_labels,
        gender_values=gender_values,
        overtime_labels=overtime_labels,
        overtime_values=overtime_values,

        # Risk
        risk_bins_labels=risk_bins_labels,
        risk_bins_values=risk_bins_values,
        risk_seg_labels=risk_seg_labels,
        risk_seg_values=risk_seg_values,

        # Feature Importance
        fi_labels=fi_labels,
        fi_values=fi_values,

        # Salary
        salary_leave_values=salary_leave,
        salary_stay_values=salary_stay,

        # Department Health
        dept_health_labels=dept_health_labels,
        dept_health_values=dept_health_values,

        # Model Comparison
        model_rows=model_rows,
        model_names=model_names,
        model_acc_values=model_acc_values,
        model_f1_values=model_f1_values,
        models_compared=models_compared,
        best_model_name=best_model_name,
        best_accuracy=best_accuracy,

        # Forecast
        forecast_labels=forecast_labels,
        forecast_values=forecast_values,

        # Text insights
        insights=insights,

        # Insights tab (KPI + charts)
        kpi_low_satisfaction=kpi_low_satisfaction,
        kpi_poor_wlb=kpi_poor_wlb,
        kpi_overtime=kpi_overtime,
        kpi_no_promo=kpi_no_promo,
        kpi_low_salary=kpi_low_salary,

        stability_labels=stability_labels,
        stability_values=stability_values,

        age_bins=age_bins,
        age_counts=age_counts,

        income_bins=income_bins,
        income_counts=income_counts,

        edu_labels=edu_labels,
        edu_values=edu_values,

        joblevel_labels=joblevel_labels,
        joblevel_values=joblevel_values,

        dept_names=dept_names,
        dept_attrition=dept_attrition,

        error=None,
    )
    

#Admin

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/admin/high-risk")
@admin_required
def high_risk_employees():
    conn = get_db()
    cur = get_cursor(conn)

    cur.execute("""
        SELECT username, risk_score
        FROM predictions
        WHERE risk_score >= 70
        ORDER BY risk_score DESC
    """)

    high_risk_list = cur.fetchall()
    conn.close()

    return render_template(
        "admin/high_risk.html",
        high_risk_list=high_risk_list
    )
    


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    conn = sqlite3.connect("users.db")
    cur = get_cursor(conn)

    # ---------------- SUMMARY CARDS ----------------
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM prediction_logs WHERE prediction_type='Single'")
    single_predictions = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM prediction_logs WHERE prediction_type='Bulk'")
    bulk_predictions = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM prediction_logs WHERE result='At Risk'")
    high_risk = cur.fetchone()[0]

    # ---------------- AT RISK vs SAFE ----------------
    cur.execute("""
        SELECT result, COUNT(*)
        FROM prediction_logs
        GROUP BY result
    """)
    r = dict(cur.fetchall())
    at_risk = r.get("At Risk", 0)
    safe = r.get("Safe", 0)

    total = at_risk + safe or 1
    at_risk_percent = round(at_risk / total * 100, 1)
    safe_percent = round(safe / total * 100, 1)

    # ---------------- PREDICTION TYPE ----------------
    cur.execute("""
        SELECT prediction_type, COUNT(*)
        FROM prediction_logs
        GROUP BY prediction_type
    """)
    p = dict(cur.fetchall())
    single = p.get("Single", 0)
    bulk = p.get("Bulk", 0)

    total_p = single + bulk or 1
    single_percent = round(single / total_p * 100, 1)
    bulk_percent = round(bulk / total_p * 100, 1)

    # ---------------- ATTRITION 12 MONTHS ----------------
    cur.execute("""
        SELECT strftime('%m', created_at) AS m,
               SUM(CASE WHEN result='At Risk' THEN 1 ELSE 0 END),
               SUM(CASE WHEN result='Safe' THEN 1 ELSE 0 END)
        FROM prediction_logs
        GROUP BY m
        ORDER BY m
    """)
    rows = cur.fetchall()

    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    risk_map = {str(i+1).zfill(2): [0,0] for i in range(12)}

    for m, r, s in rows:
        risk_map[m] = [r, s]

    at_risk_month = [risk_map[str(i+1).zfill(2)][0] for i in range(12)]
    safe_month = [risk_map[str(i+1).zfill(2)][1] for i in range(12)]

    # ---------------- TABLES ----------------
    cur.execute("""
        SELECT username, probability, department
        FROM prediction_logs
        WHERE result='At Risk'
        ORDER BY probability DESC
        LIMIT 5
    """)
    high_risk_employees = cur.fetchall()

    cur.execute("""
        SELECT username, prediction_type, result, probability, created_at
        FROM prediction_logs
        ORDER BY created_at DESC
        LIMIT 5
    """)
    prediction_logs = cur.fetchall()

    conn.close()

    return render_template(
        "admin/admin_dashboard.html",

        # cards
        total_users=total_users,
        single_predictions=single_predictions,
        bulk_predictions=bulk_predictions,
        high_risk=high_risk,

        # charts
        month_labels=month_labels,
        at_risk_month=at_risk_month,
        safe_month=safe_month,

        at_risk=at_risk,
        safe=safe,
        at_risk_percent=at_risk_percent,
        safe_percent=safe_percent,

        single=single,
        bulk=bulk,
        single_percent=single_percent,
        bulk_percent=bulk_percent,

        # tables
        high_risk_employees=high_risk_employees,
        prediction_logs=prediction_logs
    )






    
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/admin/users")
@admin_required
def admin_users():
    conn = get_db()
    cur = get_cursor(conn)
    cur.execute("SELECT id, username, email, role, status FROM users")
    users = cur.fetchall()
    conn.close()

    return render_template("admin/users.html", users=users)

from werkzeug.security import check_password_hash

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db()
        cur = get_cursor(conn)

        cur.execute("""
            SELECT id, username, password, role
            FROM users
            WHERE username = ?
        """, (username,))
        admin = cur.fetchone()
        conn.close()

        if not admin or admin["role"] != "admin":
            return render_template(
                "login.html",
                admin_login=True,
                error="Admin not found"
            )

        if not check_password_hash(admin["password"], password):
            return render_template(
                "login.html",
                admin_login=True,
                error="Invalid password"
            )

        # ✅ ADMIN LOGIN SUCCESS
        session.clear()
        session["admin_logged_in"] = True
        session["role"] = "admin"
        session["admin_username"] = admin["username"]

        return redirect(url_for("admin_dashboard"))

    # GET request
    return render_template("admin_login.html")


@app.route("/admin/reports")
@admin_required
def admin_reports():
    conn = get_db()
    cur = get_cursor(conn)
    cur.execute("""
        SELECT username, prediction, probability, created_at
        FROM predictions
        ORDER BY created_at DESC
    """)
    reports = cur.fetchall()
    conn.close()

    return render_template("admin/reports.html", reports=reports)


# ============================= DOWNLOAD ROUTES ===============================

@app.route("/employee_report/<int:emp_id>")
def employee_report(emp_id):
    global LAST_BULK_DF
    if LAST_BULK_DF is None:
        return "No bulk results available. Run a bulk prediction first."

    df = LAST_BULK_DF
    row = df[df["EmployeeNumber"] == emp_id]

    if row.empty:
        return "Employee not found."

    row = row.iloc[0]
    return render_template("employee_report.html", row=row)



# ===================== CSV (leave / stay / summary) =====================

@app.route("/download/csv/<kind>")
def download_csv_kind(kind):
    files_map = {
        "leave": "leave.csv",
        "stay": "stay.csv",
        "summary": "summary.csv"
    }

    fname = files_map.get(kind)
    if not fname:
        return "Invalid CSV type.", 400

    file_path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(file_path):
        return f"No {kind}.csv found. Run bulk first.", 404

    return send_file(file_path, as_attachment=True, download_name=f"{kind}.csv")



# ===================== EXCEL (leave / stay / summary) =====================

@app.route("/download/excel/<kind>")
def download_excel_kind(kind):
    files_map = {
        "leave": "leave.csv",
        "stay": "stay.csv",
        "summary": "summary.csv"
    }

    fname = files_map.get(kind)
    if not fname:
        return "Invalid Excel type.", 400

    csv_path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(csv_path):
        return f"No {kind}.csv found. Run bulk first.", 404

    df = pd.read_csv(csv_path)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=kind.capitalize())
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name=f"{kind}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



# ===================== PDF (leave / stay / summary) =====================

@app.route("/download/pdf/<kind>")
def download_pdf_kind(kind):
    if not REPORTLAB_AVAILABLE:
        return "Install reportlab: pip install reportlab", 500

    files_map = {
        "leave": "leave.csv",
        "stay": "stay.csv",
        "summary": "summary.csv"
    }

    fname = files_map.get(kind)
    if not fname:
        return "Invalid PDF type.", 400

    csv_path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(csv_path):
        return f"No {kind}.csv found. Run bulk first.", 404

    # read CSV with pandas and convert to PDF bytes
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading {csv_path}: {e}", 500

    title = f"{kind.capitalize()} Report"
    pdf_buf = df_to_pdf_bytes(df, title=title)

    return send_file(pdf_buf, as_attachment=True, download_name=f"{kind}.pdf", mimetype="application/pdf")


# ===================== ZIP (leave / stay / summary) =====================

@app.route("/download/zip/<kind>")
def download_zip_kind(kind):
    files_map = {
        "leave": "leave.csv",
        "stay": "stay.csv",
        "summary": "summary.csv"
    }

    fname = files_map.get(kind)
    if not fname:
        return "Invalid ZIP type.", 400

    csv_path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(csv_path):
        return f"No {kind}.csv found. Run bulk first.", 404

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading {csv_path}: {e}", 500

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        # Add CSV file (on-disk)
        try:
            zf.write(csv_path, arcname=f"{kind}.csv")
        except Exception as e:
            # fallback: write CSV from dataframe if write fails
            csv_mem = io.StringIO()
            df.to_csv(csv_mem, index=False)
            zf.writestr(f"{kind}.csv", csv_mem.getvalue().encode("utf-8"))

        # Add Excel
        try:
            excel_mem = io.BytesIO()
            with pd.ExcelWriter(excel_mem, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name=kind.capitalize())
            excel_mem.seek(0)
            zf.writestr(f"{kind}.xlsx", excel_mem.read())
        except Exception as e:
            # ignore excel if it fails
            print("Excel add to zip failed:", e)

        # Add PDF using df_to_pdf_bytes (better layout than raw canvas cell-draw)
        if REPORTLAB_AVAILABLE:
            try:
                pdf_buf = df_to_pdf_bytes(df, title=f"{kind.capitalize()} Report")
                zf.writestr(f"{kind}.pdf", pdf_buf.getvalue())
            except Exception as e:
                print("PDF add to zip failed:", e)

    zip_buffer.seek(0)
    return send_file(zip_buffer, as_attachment=True, download_name=f"{kind}.zip", mimetype="application/zip")


from flask import abort

@app.route("/employee_report/<int:emp_id>/download/<kind>")
def employee_report_download(emp_id, kind):
    if kind == "csv":
        return redirect(url_for("download_employee_csv", emp_id=emp_id))
    if kind in ("excel", "xlsx"):
        return redirect(url_for("download_employee_excel", emp_id=emp_id))
    if kind == "pdf":
        return redirect(url_for("download_employee_pdf", emp_id=emp_id))
    return abort(404)


@app.route("/download/employee/<int:emp_id>/csv")
def download_employee_csv(emp_id):
    """Download a single employee report as CSV (only the one row)."""
    global LAST_BULK_DF
    if LAST_BULK_DF is None:
        return "No bulk results available. Run a bulk prediction first.", 404

    df = LAST_BULK_DF
    row_df = df[df["EmployeeNumber"] == emp_id]
    if row_df.empty:
        return "Employee not found.", 404

    s = io.StringIO()
    row_df.to_csv(s, index=False)
    s.seek(0)
    data = s.getvalue().encode("utf-8")
    return send_file(
        io.BytesIO(data),
        as_attachment=True,
        download_name=f"employee_{emp_id}_report.csv",
        mimetype="text/csv",
    )


@app.route("/download/employee/<int:emp_id>/excel")
def download_employee_excel(emp_id):
    """Download a single employee report as XLSX (one-sheet workbook)."""
    global LAST_BULK_DF
    if LAST_BULK_DF is None:
        return "No bulk results available. Run a bulk prediction first.", 404

    df = LAST_BULK_DF
    row_df = df[df["EmployeeNumber"] == emp_id]
    if row_df.empty:
        return "Employee not found.", 404

    buf = io.BytesIO()
    try:
        # write Excel to buffer
        row_df.to_excel(buf, index=False, engine="openpyxl")
    except Exception as e:
        return f"Excel generation error: {e}", 500

    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"employee_{emp_id}_report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/download/employee/<int:emp_id>/pdf")
def download_employee_pdf(emp_id):
    """Download a single employee report as a nicely formatted PDF."""
    global LAST_BULK_DF
    if LAST_BULK_DF is None:
        return "No bulk results available. Run a bulk prediction first.", 404

    df = LAST_BULK_DF
    row_df = df[df["EmployeeNumber"] == emp_id]
    if row_df.empty:
        return "Employee not found.", 404

    row = row_df.iloc[0]

    # create PDF in memory
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left = 50
    right_col_x = 320
    y = height - 60

    # Title + badge
    p.setFont("Helvetica-Bold", 18)
    p.drawString(left, y, f"Employee Report Card — #{int(row.get('EmployeeNumber', emp_id))}")
    y -= 28

    # Horizontal line
    p.setLineWidth(0.5)
    p.line(left, y, width - 50, y)
    y -= 18

    p.setFont("Helvetica-Bold", 12)
    p.drawString(left, y, "Basic Details")
    y -= 16
    p.setFont("Helvetica", 11)

    # Basic details two-column layout
    basic_pairs = [
        ("Employee Number", int(row.get("EmployeeNumber", ""))),
        ("Age", row.get("Age", "")),
        ("Gender", row.get("Gender", "")),
        ("Marital Status", row.get("MaritalStatus", row.get("Marital_Status", ""))),
        ("Department", row.get("Department", "")),
        ("Job Role", row.get("JobRole", row.get("Job_Role", ""))),
        ("Education Field", row.get("EducationField", row.get("Education_Field", ""))),
    ]

    for label, val in basic_pairs:
        p.drawString(left, y, f"{label}:")
        p.drawString(right_col_x, y, str(val))
        y -= 16

    y -= 8
    p.setFont("Helvetica-Bold", 12)
    p.drawString(left, y, "Attrition Prediction")
    y -= 16
    p.setFont("Helvetica", 11)

    pred = row.get("AttritionPrediction", row.get("Attrition_Prediction", ""))
    prob = row.get("AttritionProbability", row.get("Attrition_Probability", ""))
    score = row.get("RiskScore", row.get("Risk_Score", ""))

    p.drawString(left, y, f"Prediction: {pred}")
    p.drawString(right_col_x, y, f"Probability: {prob}")
    y -= 18
    p.drawString(left, y, f"Risk Score: {score}")
    y -= 22

    p.setFont("Helvetica-Bold", 12)
    p.drawString(left, y, "Key Factors")
    y -= 16
    p.setFont("Helvetica", 11)

    key_pairs = [
        ("OverTime", row.get("OverTime", "")),
        ("Monthly Income", row.get("MonthlyIncome", row.get("Monthly_Income", ""))),
        ("Job Satisfaction", row.get("JobSatisfaction", row.get("Job_Satisfaction", ""))),
        ("Work Life Balance", row.get("WorkLifeBalance", row.get("Work_Life_Balance", ""))),
        ("Years at Company", row.get("YearsAtCompany", row.get("Years_At_Company", ""))),
        ("Years in Current Role", row.get("YearsInCurrentRole", row.get("Years_In_Current_Role", ""))),
        ("Years Since Last Promotion", row.get("YearsSinceLastPromotion", row.get("Years_Since_Last_Promotion", ""))),
    ]

    for label, val in key_pairs:
        p.drawString(left, y, f"{label}:")
        p.drawString(right_col_x, y, str(val))
        y -= 14
        if y < 80:
            p.showPage()
            y = height - 60

    y -= 12
    p.setFont("Helvetica-Bold", 12)
    p.drawString(left, y, "HR Recommendations")
    y -= 14
    p.setFont("Helvetica", 10)

    recs = row.get("Recommendations", "")
    # wrap text
    from reportlab.lib.utils import simpleSplit
    lines = simpleSplit(str(recs), "Helvetica", 10, width - 100)
    for line in lines:
        if y < 60:
            p.showPage()
            y = height - 60
        p.drawString(left, y, line)
        y -= 12

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name=f"employee_{emp_id}_report.pdf", mimetype="application/pdf")





# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
