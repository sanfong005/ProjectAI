import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. โหลดข้อมูล (ตัวอย่างใช้ข้อมูล SMS Spam Collection)
# Updated URL for the dataset
url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-spam-collection-dataset-/master/spam.csv"  
data = pd.read_csv(url, encoding='latin-1') # Added encoding to handle special characters

# 2. แปลงข้อความสแปมเป็น 1 และไม่เป็นสแปมเป็น 0
data['v1'] = data['v1'].map({'spam': 1, 'ham': 0}) # Changed column name to 'v1'

# 3. แยกข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42) # Changed column names to 'v2' and 'v1'

# 4. ทำการแปลงข้อความเป็นเวกเตอร์ด้วย TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. สร้างโมเดล Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. ทำการทดสอบโมเดลกับข้อมูลทดสอบ
y_pred = model.predict(X_test_tfidf)

# 7. แสดงผลลัพธ์ความแม่นยำและรายงานการจำแนกประเภท
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))