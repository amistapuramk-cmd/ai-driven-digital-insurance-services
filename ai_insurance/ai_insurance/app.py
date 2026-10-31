"""AI-Driven Digital Insurance Services - base project.
Three AI modules (trained on synthetic data so it runs out of the box):
  1. Premium prediction  (Gradient Boosting regression)
  2. Claim fraud detection (Random Forest classifier)
  3. Support chatbot      (TF-IDF intent matching)
Run:  pip install -r requirements.txt && python app.py  ->  http://127.0.0.1:5000
"""
import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
rng = np.random.default_rng(42)

# ---------- 1. Premium model (health insurance) ----------
N = 3000
age = rng.integers(18, 70, N)
bmi = rng.normal(26, 4.5, N).clip(15, 45)
smoker = rng.integers(0, 2, N)
dependents = rng.integers(0, 5, N)
coverage = rng.choice([200000, 500000, 1000000], N)
premium = (2000 + age * 90 + np.maximum(bmi - 25, 0) * 250 + smoker * 7000
           + dependents * 1200 + coverage * 0.01 + rng.normal(0, 500, N))
X_prem = np.c_[age, bmi, smoker, dependents, coverage]
premium_model = GradientBoostingRegressor().fit(X_prem, premium)

# ---------- 2. Fraud model ----------
M = 4000
claim_amt = rng.uniform(5000, 500000, M)
policy_age = rng.integers(1, 1500, M)        # days since policy start
prior_claims = rng.poisson(1, M)
report_delay = rng.integers(0, 60, M)        # days to report the claim
risk = ((claim_amt > 250000) * 1.5 + (policy_age < 90) * 2 + prior_claims * 0.6
        + (report_delay > 30) * 1.2 + rng.normal(0, 0.7, M))
is_fraud = (risk > 3.2).astype(int)
X_fraud = np.c_[claim_amt, policy_age, prior_claims, report_delay]
fraud_model = RandomForestClassifier(n_estimators=150, random_state=1).fit(X_fraud, is_fraud)

# ---------- 3. Chatbot ----------
FAQ = {
    "How do I buy a policy? Go to the Premium tab, enter your details and get an instant quote, then proceed to purchase.":
        "buy purchase new policy apply quote get insurance",
    "To file a claim, submit your policy number, incident date and documents. Claims are AI-screened and reviewed within 48 hours.":
        "file claim submit claim accident damage reimbursement",
    "Premium depends on age, BMI, smoking status, dependents and coverage amount.":
        "premium price cost how calculated why expensive",
    "Check claim status using your claim ID in the portal; you also receive SMS/email updates.":
        "claim status track where is my claim pending",
    "To renew, open My Policies and click Renew. Auto-renew can be enabled in settings.":
        "renew renewal expire expiry extend",
    "To cancel a policy, contact support with your policy number. Refunds follow the policy terms.":
        "cancel cancellation refund stop policy",
}
answers, keys = list(FAQ), list(FAQ.values())
vec = TfidfVectorizer().fit(keys)
key_mat = vec.transform(keys)

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.post("/api/premium")
def api_premium():
    d = request.get_json()
    x = [[float(d["age"]), float(d["bmi"]), int(d["smoker"]),
          int(d["dependents"]), float(d["coverage"])]]
    return jsonify(premium=round(float(premium_model.predict(x)[0]), 2))

@app.post("/api/fraud")
def api_fraud():
    d = request.get_json()
    x = [[float(d["claim_amount"]), float(d["policy_age"]),
          float(d["prior_claims"]), float(d["report_delay"])]]
    p = float(fraud_model.predict_proba(x)[0][1])
    verdict = "High risk - manual investigation" if p > .6 else \
              "Medium risk - additional documents" if p > .3 else "Low risk - fast-track approval"
    return jsonify(fraud_probability=round(p, 3), decision=verdict)

@app.post("/api/chat")
def api_chat():
    q = request.get_json().get("message", "")
    sims = cosine_similarity(vec.transform([q]), key_mat)[0]
    i = int(sims.argmax())
    reply = answers[i] if sims[i] > 0.1 else \
        "Sorry, I didn't get that. I can help with buying, claims, premiums, renewals and cancellation."
    return jsonify(reply=reply)

if __name__ == "__main__":
    app.run(debug=True)
