import cv2
import matplotlib.pyplot as plt

# Ana görüntüyü içe aktar
choes = cv2.imread(r"C:\Users\MUSA\Desktop\Nesne Tespiti\OzellikEslestirme.py\foto.jpg", 0)
plt.figure()
plt.imshow(choes, cmap="gray")
plt.axis("off")

# Aranacak olan görüntü
cho = cv2.imread(r"C:\Users\MUSA\Desktop\Nesne Tespiti\OzellikEslestirme.py\foto2.jpg", 0)
plt.figure()
plt.imshow(cho, cmap="gray")
plt.axis("off")

"""
cv2.imread(): foto.jpg ve foto2.jpg dosyalarını gri tonlamada (0 parametresi ile) okur.
Görüntüler sırasıyla ekrana yerleştirilir.
"""

# ORB tanımlayıcı
# Köşe-kenar gibi nesneye ait özellikler
orb = cv2.ORB_create()

"""
cv2.ORB_create(): ORB tanımlayıcısını oluşturur. ORB, özellikle hızlı ve etkili bir köşe ve kenar özellikleri çıkarımı için kullanılır. 
Hem anahtar nokta tespiti hem de öznitelik çıkarımı işlemleri için uygundur.
"""

# Anahtar nokta tespiti ve öznitelik çıkarımı
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(choes, None)

"""
orb.detectAndCompute(): Her iki görüntüdeki anahtar noktaları (kp1, kp2) tespit eder ve bu noktalarla ilişkili öznitelikleri (des1, des2) çıkarır.
"""

# Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Noktaları eşleştir
matches = bf.match(des1, des2)

"""
cv2.BFMatcher(): Eşleştirme işlemi için Brute Force Matcher (BFMatcher) kullanılır. cv2.NORM_HAMMING, ikili özniteliklerin karşılaştırılması için kullanılan mesafe ölçüsüdür.
bf.match(): des1 ve des2 özniteliklerini karşılaştırarak eşleşen noktaları bulur.
"""

# Mesafeye göre sırala
matches = sorted(matches, key=lambda x: x.distance)

"""
Elde edilen eşleşmeler, mesafeye göre (yani, benzerliğe göre) sıralanır. Mesafe ne kadar küçükse, eşleşme o kadar iyi kabul edilir.
"""

# Eşleşen resimleri görselleştirelim
plt.figure()
img_match = cv2.drawMatches(cho, kp1, choes, kp2, matches[:20], None, flags=2)
plt.imshow(img_match)
plt.axis("off")
plt.title("orb")

"""
cv2.drawMatches(): İlk 20 eşleşmeyi (mesafeye göre en yakın 20 eşleşme) çizerek görselleştirir.
Görüntüde eşleşen noktalar çizilir ve başlık olarak "orb" eklenir.
"""

# SIFT tanımlayıcıyı oluştur
sift = cv2.SIFT_create()

"""
cv2.SIFT_create(): SIFT tanımlayıcısını oluşturur. SIFT, ölçek ve döndürme invariyant (değişken) özellikleriyle köşe tespiti için güçlü bir yöntemdir.
"""

# Anahtar nokta tespiti ve öznitelik çıkarımı sift ile
kp1, des1 = sift.detectAndCompute(cho, None)
kp2, des2 = sift.detectAndCompute(choes, None)

"""
sift.detectAndCompute(): SIFT ile her iki görüntüdeki anahtar noktalar (kp1, kp2) ve bu noktalara karşılık gelen öznitelikler (des1, des2) çıkarılır.
"""

# Brute Force Matcher (k-nn)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

"""
bf.knnMatch(): K-en yakın komşu (k-NN) eşleştirme işlemi yapılır. Burada her bir öznitelik için en yakın iki eşleşme bulunur.
"""

guzel_eslesme = []

for match1, match2 in matches:
    if match1.distance < 0.75 * match2.distance:
        guzel_eslesme.append([match1])

"""
İyi eşleşmeleri seçmek için, birinci eşleşmenin mesafesinin ikinci eşleşmenin mesafesinin %75'inden küçük olup olmadığını kontrol eder. Bu, iyi bir eşleşme kriteridir.
"""        

# Eşleşmeleri görselleştirelim
plt.figure()
sift_matches = cv2.drawMatchesKnn(cho, kp1, choes, kp2, guzel_eslesme, None, flags=2)
plt.imshow(sift_matches)
plt.axis("off")
plt.title("SIFT ile Eşleşme")

"""
cv2.drawMatchesKnn(): K-NN eşleşmeleri görselleştirir. guzel_eslesme içindeki iyi eşleşmeleri çizerek gösterir.
"""

plt.show()
