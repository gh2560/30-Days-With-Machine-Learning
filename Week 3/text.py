from sklearn.feature_extraction.text import TfidfVectorizer

# Các văn bản mẫu
documents = ["I love programming", "Python is great for machine learning", "I enjoy coding in Python"]

# Tạo đối tượng TfidfVectorizer
vectorizer = TfidfVectorizer()

# Biến đổi văn bản thành ma trận TF-IDF
tfidf_matrix = vectorizer.fit_transform(documents)

# Hiển thị kết quả
print(tfidf_matrix.toarray())
print(vectorizer.get_feature_names_out())