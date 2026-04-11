# Decision Tree & Linear Regression (From Scratch + ML System)

##  Overview

This project is a complete Machine Learning system that implements:

* Decision Tree (ID3) from scratch
* Simple Linear Regression from scratch
* Multiple Linear Regression (bonus using sklearn)

It follows a modular pipeline-based architecture similar to industry-level ML systems.

---

##  Features

###  Decision Tree (ID3)

* Built from scratch (no sklearn)
* Entropy & Information Gain
* Recursive tree construction
* Prediction system
* JSON tree export

###  Linear Regression

* Implemented from scratch
* Uses mathematical formula for slope & intercept
* Model evaluation (MSE, R²)

### ➕ Multiple Linear Regression (Bonus)

* Uses sklearn
* Supports multiple features
* Separate pipeline

###  System Features

* Modular pipeline design
* Logging across all modules
* Model saving/loading (pickle)
* CLI-based menu system
* Visualization support

---

##  Project Structure

```
decisiontree_and_linearregression/

│
├── decision_tree/
│   ├── src/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── data_split.py
│   │   ├── entropy.py
│   │   ├── information_gain.py
│   │   ├── feature_selection.py
│   │   ├── recursive_id3.py
│   │   ├── prediction.py
│   │   ├── pipeline.py
│   │   ├── train_app.py
│   │   ├── predict_app.py
│   │   ├── model_save.py
│   │   └── visualize_tree.py
│
├── linear_regression/
│   ├── src/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── data_split.py
│   │   ├── model.py
│   │   ├── model_evaluation.py
│   │   ├── pipeline.py
│   │   ├── prediction.py
│   │   └── visualize.py
│   │
│   └── bonus_part/
│       ├── multifeature_preprocessing.py
│       ├── multifeature_model.py
│       └── multifeature_pipeline.py
│
├── main_controller.py
├── main.py
├── logs/
├── outputs/
└── artifacts/
```

---

##  How It Works

### Decision Tree Pipeline

1. Load dataset from Kaggle
2. Preprocess data (feature selection + encoding)
3. Split into train/test
4. Build tree using ID3 algorithm
5. Evaluate model (accuracy, precision, recall, F1)
6. Save model
7. Export tree as JSON

---

### Linear Regression Pipeline

1. Load dataset
2. Select features
3. Split data
4. Train model using mathematical formula
5. Evaluate (MSE, R²)
6. Save model
7. Visualize regression line

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run Main System

```bash
python main.py
```

---

### 3. Menu Options

```
1. Decision Tree
2. Linear Regression
3. Visualize Linear Regression
4. Multi Feature Linear Regression
5. Exit
```

---

### 4. Training & Prediction

* Choose model
* Select mode:

  * Train
  * Predict

---

## Sample Inputs

### Decision Tree

```python
[
    {"odor": 6, "gill-size": 1, "cap-surface": 2},
    {"odor": 3, "gill-size": 0, "cap-surface": 2}
]
```

### Linear Regression

```python
[50, 100, 150]
```

---

## Outputs

* Model files → `artifacts/`
* Logs → `logs/`
* Decision Tree JSON → `outputs/decision_tree.json`
* Regression plot → `outputs/regression_plot.png`

---

##  Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn (for bonus part)
* Matplotlib
* KaggleHub

---

##  Author

**Nouman Hafeez**

* Passionate about Machine Learning & System Design
* Focused on building production-level ML systems

---

##  Final Note

This project demonstrates:

* Strong ML fundamentals
* Clean architecture
* Pipeline-based thinking
