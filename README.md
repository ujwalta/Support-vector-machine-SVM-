# 💳 Credit Card Eligibility Prediction using SVM

This project builds a Support Vector Machine (SVM) classifier to predict whether a person is eligible for a credit card based on features like income, credit score, and debt. It includes both:

- 🧠 A Jupyter/Colab notebook for training and evaluation.
- 🌐 A Flask API to serve real-time predictions.

---

## 📁 Project Structure

📦 credit_card_svm_project
│
├── credit_card_svm.ipynb # Jupyter/Colab notebook for model training and EDA

├── model.pkl # Saved SVM model and scaler using pickle

├── app.py # Flask web API to serve predictions

├── credit_card_svm.csv # Dataset file

└── README.md # This file


---

## 🔍 Dataset Overview

The dataset contains:

- `Income`: Annual income of the user.
- 
- `Credit_Score`: Credit score (range ~300–850).
- 
- `Debt`: Total existing debt.
- 
- `Zip_Code`: Removed during preprocessing.
- 
- `Credit_Card_Eligibility`: Target variable (1 = Eligible, 0 = Not Eligible)

### ⚖️ Class Balancing

The original dataset was imbalanced. We balanced it by undersampling the majority class (class 1):

```python
ones = df[df['Credit_Card_Eligibility'] == 1].sample(n=2000)

zeroes = df[df['Credit_Card_Eligibility'] == 0]

df_balanced = pd.concat([ones, zeroes])


📊 Model Training
Algorithm: Support Vector Classifier (SVC)

Kernel: RBF

Preprocessing: StandardScaler

Train/Test Split: 80/20



svm_model = SVC(kernel='rbf', C=1, gamma='scale')

svm_model.fit(X_train, y_train)

✅ Evaluation Results

Accuracy: 100% (on balanced data)

Classification Report: Perfect precision/recall on both classes

Confusion Matrix:

![image](https://github.com/user-attachments/assets/82b7c568-b536-4c30-a6c6-e24e2ec49576)


🚀 Flask API Usage

▶️ Run the API:

python app.py

By default, it runs at http://127.0.0.1:5000/

🔗 Endpoints:

GET / – API welcome message

POST /predict – Takes JSON input and returns eligibility

📥 Sample JSON Input:



{
  "income": 55000,

  "credit_score": 690,

  "debt": 15000
}

📤 Sample JSON Output:


{

  "Credit_Card_Eligibility": "Eligible"
}

💾 Model Export

The trained model and scaler are saved using pickle:



with open("model.pkl", "wb") as f:

    pickle.dump((svm_model, scaler), f)

📷 Visualizations (Optional)

If available, upload and show screenshots like:

Confusion matrix (confusion_matrix.png)

Distribution of target classes

Any feature correlation heatmaps

🧠 Libraries Used

pandas, numpy

matplotlib, seaborn

sklearn

flask, pickle

📌 Future Improvements

Add frontend for the API

Deploy API to cloud (Heroku, Render, etc.)

Add input validation & error handling

📜 License

This project is open-source and free to use.

🙋‍♂️ Author

Ujwalta Khanal

GitHub: @ujwalta



