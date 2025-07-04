# SMS Spam Classifier

## 🙋🏽📝 Introduction
I developed a machine learning model to **classify SMS messages as either "spam" or "ham" (legitimate)**. It's a foundational exercise in Natural Language Processing (NLP) and binary classification, demonstrating the full pipeline from raw text data to a deployed-ready classification model.

## 🎯 Goal of this project
The primary goal of this project is to build an effective and robust SMS spam detection system that can:
* Preprocess raw text data for machine learning.
* Transform text into numerical features using techniques like TF-IDF.
* Train a classification model to distinguish between spam and legitimate messages.
* Evaluate the model's performance using relevant metrics beyond simple accuracy.
* Provide a working example for classifying new, unseen messages.

## 💻 Dataset
I used the **SMS Spam Collection Dataset**, sourced from the UCI Machine Learning Repository.
* It includes approximately **5,572** SMS messages.
* Each message is labeled as either `ham` (legitimate) or `spam`.
* This dataset is pretty organized and can be used to build a machine learning project.

## 🛠️ Tools used
* **Python**
* **Pandas:** For efficient data loading and manipulation.
* **NumPy:** For numerical operations.
* **NLTK (Natural Language Toolkit):** For essential text preprocessing tasks like stopwords removal.
* **Scikit-learn:** For text vectorization (TF-IDF), model training (Multinomial Naive Bayes), and comprehensive model evaluation.
* **Matplotlib:** For basic data visualization.
* **Seaborn:** For enhanced statistical data visualization.
* **Jupyter Notebook:** For interactive development and reproducible analysis.
* **Joblib:** For saving and loading trained models and vectorizers.
* **Git & GitHub:** For version control and collaborative development.

## ✨ Key Features
This project implements a complete pipeline for SMS spam classification:
* **Data Preprocessing:** Cleans raw SMS messages by converting to lowercase, removing punctuation, and filtering out common stopwords.
* **Feature Engineering:** Extracts numerical features from text using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization.
* **Machine Learning Model:** Trains a **Multinomial Naive Bayes Classifier**, a robust algorithm well-suited for text classification.
* **Model Evaluation:** Assesses classifier performance using key metrics such as **Accuracy, Precision, Recall, F1-Score, and Confusion Matrix**, paying close attention to class imbalance.
* **Model Persistence:** Saves the trained model and the TF-IDF vectorizer for future use without retraining.
* **New Message Prediction:** Demonstrates how to classify new, unseen SMS messages.

## 📊 Data Exploration & Insights
Initial exploration of the dataset revealed:
* **Class Imbalance:** The dataset is imbalanced, with higher proportion of 'ham' messages compared to 'spam'.
* **Message Length:** Another observation was the difference in message lengths; spam messages often tend to be longer due to promotional content and links.
* **No Missing Values:** The dataset has no missing values in the message content or labels.
* **Text Characteristics:** Sample messages highlighted typical patterns in spam (e.g., usage of all caps, exclamation marks, specific keywords, numerical characters) versus more casual language in ham messages.

## 🧠 Model & Methodology
The classification pipeline follows these steps:
1.  **Text Preprocessing:** Each raw SMS message is cleaned by:
    * Converting to lowercase.
    * Removing all punctuation.
    * Removing common English stopwords (e.g., 'the', 'is', 'and').
    The result is a list of clean, meaningful words.
2.  **Feature Vectorization:** The cleaned text tokens are transformed into a numerical format using `TfidfVectorizer`. This creates a matrix where each row represents a message and columns represent words, with values indicating the TF-IDF score of each word.
3.  **Data Splitting:** The dataset is split into 80% training data and 20% testing data, using `stratify=y` to maintain the original class distribution in both sets, crucial for handling imbalance.
4.  **Model Training:** A **Multinomial Naive Bayes Classifier** is trained on the preprocessed and vectorized training data. This model is pretty good for classification tasks with count-based or frequency-based features like TF-IDF.
5.  **Prediction & Evaluation:** The trained model predicts labels on the unseen test set. Performance is rigorously evaluated using Accuracy, Confusion Matrix, Precision, Recall, and F1-Score.
6.  **Model Saving:** The trained `MultinomialNB` model and the `TfidfVectorizer` are saved using `joblib` for future inference without retraining.

## 📈 Evaluation Metrics & Results

The model's performance was evaluated on the unseen test set, yielding the following results:
* **Accuracy:** 0.9704
* **Confusion Matrix:**
    ```
    [[966, 0],
     [33, 116]]
    ```
    * **True Negatives (TN):** Correctly classified Ham messages.
    * **False Positives (FP):** Ham messages incorrectly classified as Spam (Type I Error - important to minimize).
    * **False Negatives (FN):** Spam messages incorrectly classified as Ham (Type II Error - important to minimize).
    * **True Positives (TP):** Correctly classified Spam messages.
* **Classification Report:**

    ```
              precision    recall  f1-score   support

         ham       0.97      1.00      0.98       966
        spam       1.00      0.78      0.88       149

    accuracy                           0.97      1115
   macro avg       0.98      0.89      0.93      1115
weighted avg       0.97      0.97      0.97      1115
    ```
* **Key Observations:** The model shows strong performance, especially with a high [e.g., precision for spam, or F1-score] indicating its ability to identifying spam messages while minimizing false positives. The class imbalance was handled well by leveraging `stratify` in the train-test split.

## 🚀 How to Run Locally
To set up and run this project on your local machine:
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/sankethaabhishek/sms-spam-classifier.git](https://github.com/sankethaabhishek/sms-spam-classifier.git)
    cd sms-spam-classifier
    ```
2.  **Download Dataset:**
    * Download the `smsspamcollection.zip` dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip).
    * Extract the contents.
    * Copy the `SMSSpamCollection` file into the `data/raw/` directory within your cloned repository.
        ```
        sms-spam-classifier/
        └── data/
            └── raw/
                └── SMSSpamCollection
        ```

3.  **Set up Python Environment & Install Libraries:**
    * Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
    * Open your terminal and navigate to the project root (`sms-spam-classifier/`).
    * Install the required libraries:
        ```bash
        conda install jupyter pandas numpy scikit-learn matplotlib seaborn nltk joblib
        ```
        (If you prefer pip: `pip install jupyter pandas numpy scikit-learn matplotlib seaborn nltk joblib`)
    * **Download NLTK Data (if prompted during notebook execution):**
        When running the notebook, `nltk` might prompt you to download `stopwords` and `wordnet`. If so, run these in a Jupyter cell:
        ```python
        import nltk
        nltk.download('stopwords')
        nltk.download('wordnet')
        ```

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    This will open the Jupyter interface in your web browser.

5.  **Explore the Notebook:**
    * Navigate to the `notebooks/` folder.
    * Open and run all cells in `01_sms_spam_classifier_full_pipeline.ipynb` sequentially to see the entire process, from data loading to model prediction.

## 🚀 Future Enhancements
Possible future improvements for this project include:
* **Advanced NLP Techniques:** Explore word embeddings (Word2Vec, GloVe), or transformer models for more sophisticated text representation.
* **Deep Learning Models:** Experiment with Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs) for text classification.
* **Hyperparameter Tuning:** Systematically optimize model parameters using GridSearchCV or RandomizedSearchCV.
* **More Robust Evaluation:** Implement cross-validation for more reliable performance estimates.
* **Deployment:** Build a simple web application (e.g., using Flask/Streamlit) to classify messages in real-time.
* **Handling Imbalance:** Investigate techniques like oversampling (SMOTE) or undersampling to manage the class imbalance more explicitly.

## 📧 Author
* **Abhishek Waduge** - [GitHub Profile](https://github.com/sankethaabhishek) | [LinkedIn Profile](https://www.linkedin.com/in/sankethaabhishekbiz)
Hit me up for any questions!
