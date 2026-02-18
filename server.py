from flask import Flask, request, jsonify, render_template, redirect, session, url_for, send_file
import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import os
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------
# 1. REAL ESTATE SYSTEM LOGIC (ADVANCED CORE)
# ---------------------------------------------------------
class RealEstateManager:
    """Handles AI Inference, Data Analytics, and Visualizations."""
    def __init__(self):
        # Paths must match your folder structure
        self.model_path = "models/house_model.pkl"
        self.cols_path = "models/columns.json"
        self.data_path = "data/processed_data.csv"
        
        # Validation
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("AI Model not found. Please run model.py first.")

        # Load AI Artifacts
        self.model = joblib.load(self.model_path)
        with open(self.cols_path, "r") as f:
            self.data_columns = json.load(f)["data_columns"]
        
        # Load Dataset for recommendations and analytics
        self.df = pd.read_csv(self.data_path)
        
        # Extract location names (skipping the first 10 numeric features)
        self.locations = [c for c in self.data_columns if c not in [
            'total_sqft','bath','bhk','age','floor_no','total_floors','parking','lift','pool', 'luxury_score'
        ]]

    def predict_price(self, form_data):
        """Processes input and returns AI-calculated market price."""
        x = np.zeros(len(self.data_columns))
        
        # Feature Mapping
        x[0] = float(form_data.get('sqft', 0))
        x[1] = int(form_data.get('bath', 1))
        x[2] = int(form_data.get('bhk', 1))
        x[3] = int(form_data.get('age', 0))
        x[4] = int(form_data.get('floor', 1))
        x[5] = int(form_data.get('total_floors', 1))
        
        # Amenities
        p = 1 if form_data.get('parking') else 0
        l = 1 if form_data.get('lift') else 0
        s = 1 if form_data.get('pool') else 0
        x[6], x[7], x[8] = p, l, s
        
        # Derived Feature used during training
        x[9] = p + l + s # Luxury Score

        # Location (One-Hot Encoding)
        loc = form_data.get('location', '').lower()
        if loc in self.data_columns:
            x[self.data_columns.index(loc)] = 1

        prediction = self.model.predict([x])[0]
        return round(max(0, prediction), 2)

    def generate_market_plot(self, location):
        """Generates a distribution plot of prices in a specific area."""
        plt.figure(figsize=(6, 4))
        # Ensure 'location' column exists in your CSV and is lowercase
        area_prices = self.df[self.df['location'].str.lower() == location.lower()]['price_lakhs']
        
        if not area_prices.empty:
            sns.histplot(area_prices, kde=True, color='#00d2ff')
            plt.title(f"Price Pulse: {location.capitalize()}")
            plt.xlabel("Lakhs")
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url

# ---------------------------------------------------------
# 2. FLASK APP & DATABASE CONFIG
# ---------------------------------------------------------
app = Flask(__name__)
app.secret_key = "Ahmedabad_Skyline_2026"
re_system = RealEstateManager()

def get_db():
    conn = sqlite3.connect("real_estate_data.db")
    conn.row_factory = sqlite3.Row
    return conn

# Ensure Database is ready
with get_db() as con:
    con.execute("""
        CREATE TABLE IF NOT EXISTS history (
            user TEXT, location TEXT, bhk INTEGER, 
            sqft REAL, price REAL, date TEXT
        )
    """)

# ---------------------------------------------------------
# 3. ROUTES
# ---------------------------------------------------------

@app.route('/')
def index():
    return render_template("landing.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['user'] = request.form.get('username')
        return redirect(url_for('dashboard'))
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("dashboard.html", locations=re_system.locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        price = re_system.predict_price(request.form)
        # Log search to history
        with get_db() as con:
            con.execute("INSERT INTO history VALUES (?,?,?,?,?,?)", (
                session.get('user', 'Guest'),
                request.form.get('location'),
                request.form.get('bhk'),
                request.form.get('sqft'),
                price,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ))
        return jsonify({"estimated_price_lakhs": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/analyze-investment', methods=['POST'])
def analyze_investment():
    """Returns Prediction + Market Analytics."""
    try:
        price = re_system.predict_price(request.form)
        loc = request.form.get('location').lower()
        
        # Investment Logic comparing to area average
        area_avg = round(re_system.df[re_system.df['location'].str.lower() == loc]['price_lakhs'].mean(), 2)
        score = 85 if price < area_avg else 60
        
        return jsonify({
            "predicted_price": price,
            "analysis": {
                "score": score,
                "rating": "High Growth" if score > 80 else "Stable",
                "area_avg": area_avg
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/market-pulse/<location>')
def market_pulse(location):
    """Sends a Base64 encoded chart of the area."""
    try:
        plot_data = re_system.generate_market_plot(location)
        return jsonify({"visualization": plot_data})
    except:
        return jsonify({"error": "Plot failed"}), 404

@app.route('/api/export-valuation', methods=['POST'])
def export_valuation():
    """Generates and serves a Valuation Report."""
    price = re_system.predict_price(request.form)
    report = f"""
    OFFICIAL VALUATION REPORT
    -------------------------
    Date: {datetime.now().strftime('%Y-%m-%d')}
    Location: {request.form.get('location').upper()}
    BHK: {request.form.get('bhk')} | Area: {request.form.get('sqft')} sqft
    
    ESTIMATED MARKET VALUE: INR {price} LAKHS
    """
    return send_file(io.BytesIO(report.encode()), mimetype='text/plain', as_attachment=True, download_name="Valuation_Report.txt")

@app.route('/recommend')
def recommend():
    """Returns top 10 properties in the location closest to the predicted price."""
    try:
        loc = request.args.get('location', '').lower()
        predicted_price = float(request.args.get('price', 0))
        
        # 1. Filter properties in the same location using the dataframe in re_system
        subset = re_system.df[re_system.df['location'].str.lower() == loc].copy()
        
        if subset.empty:
            return jsonify([])

        # 2. Calculate the "Price Distance"
        subset['price_diff'] = abs(subset['price_lakhs'] - predicted_price)
        
        # 3. Sort by the smallest difference and take exactly 10
        top_10 = subset.sort_values(by='price_diff').head(10)

        return jsonify(
            top_10[['projectName', 'price_lakhs', 'total_sqft', 'bhk']].to_dict(orient='records')
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(port=5000, debug=True)