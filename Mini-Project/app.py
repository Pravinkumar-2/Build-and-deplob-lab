import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         login_required, logout_user, current_user)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

model = joblib.load('model.pkl')          # trained Linear Regression model
FEATURES = ['Temperature', 'DayOfWeek', 'Month', 'Lag_1', 'Lag_7']

# --- Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    gender = db.Column(db.String(20))
    age = db.Column(db.Integer)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Routes ---
@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        if User.query.filter_by(username=username).first():
            flash('User already exists', 'error')
            return redirect(url_for('register'))
        user = User(username=username,
                    gender=request.form['gender'],
                    age=int(request.form['age']),
                    password=generate_password_hash(request.form['password']))
        db.session.add(user)
        db.session.commit()
        flash('Registered successfully', 'success')
        return redirect(url_for('login'))
    return render_template('Register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials', 'error')
    return render_template('Login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction_text = ""
    if request.method == 'POST':
        values = [[float(request.form['Temperature']),
                   int(request.form['DayOfWeek']),
                   int(request.form['Month']),
                   float(request.form['Lag_1']),
                   float(request.form['Lag_7'])]]
        row = pd.DataFrame(values, columns=FEATURES)
        predicted = model.predict(row)[0]
        prediction_text = f"Predicted Electricity Consumption: {predicted:.2f} kWh"
    return render_template('Index.html', username=current_user.username,
                           prediction_text=prediction_text)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
