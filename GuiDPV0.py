import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ฟังก์ชันสำหรับการประมวลผลและฝึกสอนโมเดล
def train_spam_filter():
    # โหลดข้อมูล (ใช้ชุดข้อมูล SMS Spam Collection)
    url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-spam-collection-dataset-/master/spam.csv"
    data = pd.read_csv(url, encoding='latin-1')
    
    # แปลงข้อความสแปมเป็น 1 และไม่เป็นสแปมเป็น 0
    data['v1'] = data['v1'].map({'spam': 1, 'ham': 0})
    
    # แยกข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)
    
    # ทำการแปลงข้อความเป็นเวกเตอร์ด้วย TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # สร้างโมเดล Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # ทดสอบโมเดล
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    return model, vectorizer

# ฝึกสอนโมเดล
model, vectorizer = train_spam_filter()

# ฟังก์ชันสำหรับการสแกนข้อความสแปม
def scan_spam():
    message = message_entry.get("1.0", tk.END).strip()
    
    if not message:
        result_text.insert(tk.END, "กรุณาใส่ข้อความที่ต้องการตรวจสอบ\n")
        return

    # แปลงข้อความเป็น TF-IDF เวกเตอร์
    message_tfidf = vectorizer.transform([message])
    
    # ทำนายว่าข้อความเป็นสแปมหรือไม่
    prediction = model.predict(message_tfidf)
    
    # แสดงผลลัพธ์
    if prediction == 1:
        result_text.insert(tk.END, "ข้อความนี้เป็นสแปม\n")
    else:
        result_text.insert(tk.END, "ข้อความนี้ไม่เป็นสแปม\n")

# ฟังก์ชันสำหรับการล้างผลลัพธ์
def clear_results():
    result_text.delete("1.0", tk.END)

# ฟังก์ชันสำหรับการแสดงข้อความ About
def show_about():
    messagebox.showinfo("About", "โปรแกรมนี้เป็นเครื่องมือสำหรับการกรองข้อความสแปม")

# สร้างหน้าต่างหลัก
root = tk.Tk()
root.title("Spam Filter Tool")
root.geometry("600x400")

# กรอบข้อความ
message_frame = tk.Frame(root)
message_frame.pack(pady=10)

message_label = tk.Label(message_frame, text="ใส่ข้อความที่ต้องการตรวจสอบ:")
message_label.pack()

message_entry = tk.Text(message_frame, height=5, width=50)
message_entry.pack()

scan_spam_button = tk.Button(message_frame, text="ตรวจสอบข้อความ", command=scan_spam)
scan_spam_button.pack(pady=5)

# แสดงผลลัพธ์การสแกน
result_frame = tk.Frame(root)
result_frame.pack(pady=10)

result_label = tk.Label(result_frame, text="ผลการสแกน:")
result_label.pack()

result_text = tk.Text(result_frame, height=10, width=70)
result_text.pack()

# ปุ่มล้างผลลัพธ์
clear_button = tk.Button(root, text="ล้างผลลัพธ์", command=clear_results)
clear_button.pack(pady=5)

# เมนู About
menu = tk.Menu(root)
root.config(menu=menu)

help_menu = tk.Menu(menu)
menu.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="About", command=show_about)

root.mainloop()
