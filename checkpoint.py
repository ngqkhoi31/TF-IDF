import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Bước 1: Đọc dữ liệu từ file JSON

with open(r"D:\University\Fourth-year Collage\Clould Computing\AI_Search_TF IDF\Test TF_IDF\dataset\data.json", encoding='utf-8') as file:
    data = json.load(file)


# Bước 2: Chuẩn bị dữ liệu (lấy tiêu đề và mô tả sản phẩm)
documents = []
for item in data:
    title = item.get("title", "")
    description = item.get("desc_2", "")
    documents.append(f"{title} {description}")

# Bước 3: Tính TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Bước 4: Hàm tìm kiếm sản phẩm dựa trên query
def search_products(query, top_n=5):
    # Tính TF-IDF cho truy vấn
    query_vec = vectorizer.transform([query])
    
    # Tính độ tương đồng cosine giữa truy vấn và tất cả các tài liệu
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Lấy top N sản phẩm có độ tương đồng cao nhất
    top_indices = similarity.argsort()[-top_n:][::-1]
    
    # Trả về các sản phẩm tương ứng
    results = []
    for index in top_indices:
        product = data[index]
        results.append({
            "title": product.get("title", ""),
            "price": product.get("price", ""),
            "description": product.get("desc_2", ""),
            "similarity": similarity[index]
        })
    return results

# Bước 5: Thử nghiệm tìm kiếm
query = "áo nữ thời trang kiểu nữ hàn quốc"
results = search_products(query)

# In kết quả
for result in results:
    print(f"Title: {result['title']}")
    print(f"Price: {result['price']}")
    print(f"Description: {result['description']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print("="*50)
