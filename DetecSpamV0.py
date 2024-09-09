import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. โหลดข้อมูล (ตัวอย่างใช้ข้อมูล SMS Spam Collection)
url = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# 2. แปลงข้อความสแปมเป็น 1 และไม่เป็นสแปมเป็น 0
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# 3. แยกข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

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
