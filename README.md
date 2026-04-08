#  Waste Classifier (ML Project)

##  Overview

This project is a Machine Learning-based web application that classifies waste into different categories using image classification. It helps in promoting proper waste segregation and environmental sustainability.

---

##  Features

* Upload an image of waste
* Predicts the category of waste
* Simple and interactive UI using Streamlit
* Fast and efficient prediction

---

##  Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* NumPy, OpenCV

---

##  Project Structure

```
waste-classifier/
│
├── app/              # Streamlit app
├── src/              # Model & training code
├── models/           # Trained model (ignored)
├── data/             # Dataset (ignored)
├── .gitignore
├── README.md
└── requirements.txt
```

---

##  How to Run Locally

1. Clone the repository:

```
git clone https://github.com/SakshiJhinjhore/waste-classifier.git
cd waste-classifier
```

2. Create virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the app:

```
streamlit run app/app.py
```

---

##  Note

* Dataset and trained model files are not included due to size limitations.
* You can train the model using your own dataset.

---

##  Future Improvements

* Add more waste categories
* Improve model accuracy
* Deploy as a web application
* Add real-time camera detection

---

##  Author

Sakshi Jhinjhore
