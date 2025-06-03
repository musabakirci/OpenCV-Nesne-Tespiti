import cv2
import os

img_dir = "YayaTakibi"  # Görsellerin bulunduğu dizini belirt
files = os.listdir(img_dir)
img_path_list = []

# .jpg dosyalarını listele
for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(os.path.join(img_dir, f))  # Tam yol ekle

print(img_path_list)

# HOG tanımlayıcısı oluştur
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Resimleri oku ve göster
for imagePath in img_path_list:
    print(imagePath)

    image = cv2.imread(imagePath)

    if image is None:
        print(f"Resim yüklenemedi: {imagePath}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # İnsanları tespit et
    (rects, weights) = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Tespit edilen alanları çiz
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Görüntüyü göster
    cv2.imshow("Image", image)

    # 'q' tuşuna basılana kadar bekle
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()  # Pencereleri kapat


"""
HOG Algoritmasinin Genel Adimlari:
Goruntuyu yukle ve gri tonlamaya donustur.
Gradient hesapla (Yatay ve dikey egilimler ile buyukluk).
Hucrelere bol ve histogramlari olustur.
Bloklari normalleştir.
Ozellik vektorunu olustur.
Siniflandirici ile tespit yap.
"""

