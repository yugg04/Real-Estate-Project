
# Real-Estate-Project

A Machine Learning–based Real Estate Price Prediction web application developed using Flask.
This project predicts house prices based on user inputs and historical housing data.

---

## 📌 Features

* Predicts real estate prices using trained ML model
* User-friendly web interface
* Data preprocessing and cleaning
* Model training and evaluation
* CSV dataset support
* Flask-based backend

---

## 🗂 Project Structure

```
Real-Estate-Project/
│
├── data/
│   └── processed_data.csv
│
├── models/
│   └── trained_model.pkl
│
├── static/
│   └── css/
│
├── templates/
│   └── index.html
│
├── model.py
├── server.py
├── ahmedabad_housing.csv
├── real_estate_data.db
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* Flask
* NumPy
* Pandas
* Scikit-learn
* HTML
* CSS

---

## 🚀 Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/yugg04/Real-Estate-Project.git
```

### Step 2: Enter Project Folder

```bash
cd Real-Estate-Project
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

*(Create `requirements.txt` if not available)*

### Step 4: Run Application

```bash
python server.py
```

### Step 5: Open in Browser

```
http://127.0.0.1:5000/
```

---

## 📊 How the System Works

1. Raw housing data is collected.
2. Data is cleaned and preprocessed.
3. Features are selected.
4. Machine learning model is trained.
5. Model is saved in `models/` folder.
6. Flask loads the model.
7. User enters property details.
8. System predicts house price.

---

## 🧠 Machine Learning Model

The project uses regression techniques for prediction.

Main steps:

* Handling missing values
* Feature encoding
* Data normalization
* Model training
* Performance evaluation

*(You may specify the exact algorithm here: Linear Regression, Random Forest, etc.)*

---

## 📁 Dataset

The dataset contains information such as:

* Property location
* Area (sq.ft)
* Number of bedrooms
* Number of bathrooms
* Price
* Other features

Source: Ahmedabad Housing Dataset (CSV)

---

## 💻 Usage

1. Start the Flask server.
2. Open the website in browser.
3. Enter property details.
4. Click **Predict**.
5. View estimated price.

---

## ⚠️ Limitations

* Accuracy depends on dataset quality.
* Works best for limited locations.
* Not suitable for real commercial use.
* Requires more real-time data.

---

## 🔮 Future Improvements

* Add real-time market data
* Improve UI design
* Use deep learning models
* Add user authentication
* Deploy on cloud (AWS / Heroku)

---

## 📜 License

This project is developed for educational purposes only.

---

## 👨‍💻 Author

Yug Khatri
GitHub: [https://github.com/yugg04](https://github.com/yugg04)


