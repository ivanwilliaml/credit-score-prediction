import re
import sys
import __main__
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

# Trik agar pickle bisa di-load: pkl disimpan dari module "train", di sini class yang sama didefinisikan ulang di __main__
sys.modules["train"] = __main__

# Konstanta preprocessing
ORDINAL_MAPS = {
    "Month": {m: i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August"])},
    "Credit_Mix": {"Bad": 0, "Standard": 1, "Good": 2, "Unknown": 1},
    "Spending_Level": {"Low": 0, "High": 1, "Unknown": -1},
    "Payment_Size": {"Small": 0, "Medium": 1, "Large": 2, "Unknown": -1},
    "Payment_of_Min_Amount": {"No": 0, "Yes": 1, "Unknown": 0},
}
IQR_COLS = ["Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan", "Num_of_Delayed_Payment",
            "Num_Credit_Inquiries", "Delay_from_due_date", "Interest_Rate"]
P99_COLS = ["Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt",
            "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]
ENGINEERED = ["Debt_to_Income", "EMI_to_Salary", "Investment_Rate", "Balance_to_Salary",
              "Credit_History_Years", "Loan_per_Account", "Delay_per_Loan", "Debt_per_Card",
              "Total_Credit_Products", "Delay_to_History"]
HIGH_CARD = ["Occupation", "Primary_Loan"]


# Custom transformer
# Duplikat persis dari class di train.py (wajib ada di sini agar joblib.load bisa unpickle pipeline-nya)
class CreditScorePreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing notebook (clean → impute → cap → decompose → FE → ordinal + one-hot)."""

    def _core_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data = data.drop(columns=["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"], errors="ignore")
        for col in ["Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
                    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
                    "Monthly_Balance"]:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace("", np.nan),
                errors="coerce")
        data["Age"] = data["Age"].where(data["Age"].between(18, 100), np.nan)
        data["Monthly_Balance"] = data["Monthly_Balance"].where(data["Monthly_Balance"] >= 0, np.nan)
        for col in ["Num_of_Loan", "Num_Bank_Accounts", "Num_of_Delayed_Payment",
                    "Delay_from_due_date", "Num_Credit_Card", "Num_Credit_Inquiries"]:
            data[col] = pd.to_numeric(data[col], errors="coerce").clip(lower=0)

        def to_months(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return float(x)
            m = re.match(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(x))
            return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan
        data["Credit_History_Age"] = data["Credit_History_Age"].apply(to_months)

        data["Occupation"] = data["Occupation"].replace("_______", "Unknown")
        data["Credit_Mix"] = data["Credit_Mix"].replace("_", "Unknown")
        data["Payment_of_Min_Amount"] = data["Payment_of_Min_Amount"].replace("NM", "Unknown")
        data["Payment_Behaviour"] = data["Payment_Behaviour"].replace("!@9#%8", np.nan)
        data["Type_of_Loan"] = data["Type_of_Loan"].fillna("Not Specified")
        return data

    def _decompose(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["Spending_Level"] = data["Payment_Behaviour"].str.extract(r"(Low|High)_spent")
        data["Payment_Size"] = data["Payment_Behaviour"].str.extract(r"(Small|Medium|Large)_value")
        data[["Spending_Level", "Payment_Size"]] = data[["Spending_Level", "Payment_Size"]].fillna("Unknown")
        data = data.drop(columns=["Payment_Behaviour"])
        loans = data["Type_of_Loan"].str.replace(" and ", ",", regex=False)
        data["Num_Loan_Types"] = loans.apply(
            lambda s: 0 if str(s).strip() == "Not Specified" else len([t for t in str(s).split(",") if t.strip()]))
        data["Primary_Loan"] = loans.str.split(",").str[0].str.strip()
        data = data.drop(columns=["Type_of_Loan"])
        return data

    def _add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        monthly_income = data["Annual_Income"] / 12 + 1
        data["Debt_to_Income"] = data["Outstanding_Debt"] / monthly_income
        data["EMI_to_Salary"] = data["Total_EMI_per_month"] / (data["Monthly_Inhand_Salary"] + 1)
        data["Investment_Rate"] = data["Amount_invested_monthly"] / (data["Monthly_Inhand_Salary"] + 1)
        data["Balance_to_Salary"] = data["Monthly_Balance"] / (data["Monthly_Inhand_Salary"] + 1)
        data["Credit_History_Years"] = data["Credit_History_Age"] / 12
        data["Loan_per_Account"] = data["Num_of_Loan"] / (data["Num_Bank_Accounts"] + 1)
        data["Delay_per_Loan"] = data["Num_of_Delayed_Payment"] / (data["Num_of_Loan"] + 1)
        data["Debt_per_Card"] = data["Outstanding_Debt"] / (data["Num_Credit_Card"] + 1)
        data["Total_Credit_Products"] = data["Num_Bank_Accounts"] + data["Num_Credit_Card"] + data["Num_of_Loan"]
        data["Delay_to_History"] = data["Num_of_Delayed_Payment"] / (data["Credit_History_Age"] / 12 + 1)
        return data

    def _apply_ordinal(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for col, mp in ORDINAL_MAPS.items():
            data[col] = data[col].map(mp).fillna(-1).astype(float)
        return data

    def fit(self, X, y=None):
        df = self._core_clean(X)
        self.num_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = df.select_dtypes(include=["object"]).columns.tolist()
        self.medians_ = {c: df[c].median() for c in self.num_cols_}
        df[self.num_cols_] = df[self.num_cols_].fillna(self.medians_)
        df[self.cat_cols_] = df[self.cat_cols_].fillna("Unknown")
        self.caps_ = {}
        for col in IQR_COLS:
            q1, q3 = df[col].quantile([0.25, 0.75]); iqr = q3 - q1
            self.caps_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        for col in P99_COLS:
            self.caps_[col] = (df[col].min(), df[col].quantile(0.99))
        for col, (lo, hi) in self.caps_.items():
            df[col] = df[col].clip(lo, hi)
        df = self._add_features(self._decompose(df))
        for col in ENGINEERED:
            hi = df[col].quantile(0.99)
            self.caps_[col] = (df[col].min(), hi)
            df[col] = df[col].clip(upper=hi)
        df = self._apply_ordinal(df)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[HIGH_CARD])
        self.ohe_names_ = self.ohe_.get_feature_names_out(HIGH_CARD).tolist()
        self.feature_names_ = df.drop(columns=HIGH_CARD).columns.tolist() + self.ohe_names_
        return self

    def transform(self, X) -> pd.DataFrame:
        df = self._core_clean(X)
        df[self.num_cols_] = df[self.num_cols_].fillna(self.medians_)
        df[self.cat_cols_] = df[self.cat_cols_].fillna("Unknown")
        for col in IQR_COLS + P99_COLS:
            lo, hi = self.caps_[col]
            df[col] = df[col].clip(lo, hi)
        df = self._add_features(self._decompose(df))
        for col in ENGINEERED:
            df[col] = df[col].clip(upper=self.caps_[col][1])
        df = self._apply_ordinal(df)
        ohe_arr = self.ohe_.transform(df[HIGH_CARD])
        out = pd.concat([df.drop(columns=HIGH_CARD),
                         pd.DataFrame(ohe_arr, columns=self.ohe_names_, index=df.index)], axis=1)
        return out.reindex(columns=self.feature_names_, fill_value=0)


# Load model sekali saja & cache (supaya tidak reload pkl tiap ada interaksi user)
@st.cache_resource
def load_model():
    pkl_path = Path(__file__).parent / "artifacts" / "credit_score_pipeline.pkl"
    return joblib.load(pkl_path)

model = load_model()
CLASSES = list(model.classes_)                       # ['Good', 'Poor', 'Standard']

# Skema kolom mentah yang dibutuhkan pipeline
COLUMNS_ORDER = ["Month", "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary","Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
                 "Type_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment","Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
                 "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount","Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance"]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August"]
OCCUPATIONS = ["Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", "Lawyer","Media_Manager", "Doctor", "Journalist", "Manager", "Accountant", "Musician","Mechanic", "Writer", "Architect", "Unknown"]
LOAN_TYPES = ["Not Specified", "Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan","Mortgage Loan", "Student Loan", "Debt Consolidation Loan", "Payday Loan"]

# Page
st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="wide")
st.title("💳 Credit Score Classification")
st.markdown("Memprediksi performa kredit nasabah — **Poor / Standard / Good** — dengan model LightGBM.")
st.markdown("---")


# Jalankan prediksi pada 1 input form & render hasil (label + confidence + progress bar tiap kelas)
def predict_and_show(features: dict, header: str):
    """Menjalankan inferencing pada satu record mentah & menampilkan hasil."""
    df = pd.DataFrame([features], columns=COLUMNS_ORDER)
    pred = str(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    prob_map = {c: float(p) for c, p in zip(CLASSES, proba)}

    st.subheader(header)
    msg = f"Credit Score: {pred}  |  Confidence: {prob_map[pred]*100:.1f}%"
    if pred == "Good":
        st.success(msg)
    elif pred == "Poor":
        st.error(msg)
    else:
        st.warning(msg)

    st.write("**Probability Distribution**")
    for c in ["Poor", "Standard", "Good"]:
        st.caption(f"{c}  —  {prob_map[c]*100:.1f}%")
        st.progress(float(prob_map[c]))


# ---------------- Manual input form ----------------
st.subheader("📝 Input Data Nasabah")
with st.form("manual_form"):

    # 1. Profil Nasabah
    st.markdown("#### 👤 Profil Nasabah")
    c1, c2, c3 = st.columns(3)
    with c1:
        month = st.selectbox("Month", MONTHS)
    with c2:
        age = st.number_input("Age", 18, 100, 35)
    with c3:
        occupation = st.selectbox("Occupation", OCCUPATIONS)

    st.divider()

    # 2. Pendapatan & Pengeluaran
    st.markdown("#### 💰 Pendapatan & Pengeluaran")
    c1, c2, c3 = st.columns(3)
    with c1:
        annual_income = st.number_input("Annual Income (USD)", 0.0, 300000.0, 40000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary (USD)", 0.0, 30000.0, 3500.0)
    with c2:
        total_emi = st.number_input("Total EMI per Month (USD)", 0.0, 5000.0, 70.0)
        invested = st.number_input("Amount Invested Monthly (USD)", 0.0, 10000.0, 150.0)
    with c3:
        balance = st.number_input("Monthly Balance (USD)", 0.0, 2000.0, 350.0)

    st.divider()

    # 3. Akun & Riwayat Kredit 
    st.markdown("#### 🏦 Akun & Riwayat Kredit")
    c1, c2, c3 = st.columns(3)
    with c1:
        num_bank = st.number_input("Num Bank Accounts", 0, 20, 5)
        num_card = st.number_input("Num Credit Card", 0, 20, 5)
        num_loan = st.number_input("Num of Loan", 0, 15, 3)
        loan_type = st.selectbox("Type of Loan (primary)", LOAN_TYPES)
    with c2:
        interest = st.number_input("Interest Rate (%)", 0, 50, 14)
        outstanding = st.number_input("Outstanding Debt (USD)", 0.0, 5000.0, 1200.0)
        util = st.number_input("Credit Utilization Ratio (%)", 0.0, 60.0, 32.0)
        hist_age = st.number_input("Credit History Age (months)", 0, 500, 220)
    with c3:
        changed_limit = st.number_input("Changed Credit Limit", -10.0, 50.0, 9.0)
        num_inquiries = st.number_input("Num Credit Inquiries", 0, 50, 6)
        credit_mix = st.radio("Credit Mix", ["Bad", "Standard", "Good", "Unknown"])

    st.divider()

    # 4. Perilaku Pembayaran 
    st.markdown("#### 💳 Perilaku Pembayaran")
    c1, c2, c3 = st.columns(3)
    with c1:
        delay_due = st.number_input("Delay from Due Date (days)", 0, 100, 20)
        num_delayed = st.number_input("Num of Delayed Payment", 0, 50, 12)
    with c2:
        pay_min = st.radio("Payment of Min Amount", ["Yes", "No"], horizontal=True)
        spending = st.radio("Spending Level", ["Low", "High"], horizontal=True)
    with c3:
        pay_size = st.radio("Payment Size", ["Small", "Medium", "Large"], horizontal=True)

    st.divider()
    submitted = st.form_submit_button("🔮 Prediksi Credit Score", use_container_width=True)

if submitted:
    features = {
        "Month": month, "Age": age, "Occupation": occupation, "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary, "Num_Bank_Accounts": num_bank,
        "Num_Credit_Card": num_card, "Interest_Rate": interest, "Num_of_Loan": num_loan,
        "Type_of_Loan": loan_type, "Delay_from_due_date": delay_due,
        "Num_of_Delayed_Payment": num_delayed, "Changed_Credit_Limit": changed_limit,
        "Num_Credit_Inquiries": num_inquiries, "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding, "Credit_Utilization_Ratio": util,
        "Credit_History_Age": hist_age, "Payment_of_Min_Amount": pay_min,
        "Total_EMI_per_month": total_emi, "Amount_invested_monthly": invested,
        "Payment_Behaviour": f"{spending}_spent_{pay_size}_value_payments", "Monthly_Balance": balance,
    }
    predict_and_show(features, "Hasil Prediksi")
