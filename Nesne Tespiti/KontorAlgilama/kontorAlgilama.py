import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resmi içe aktar
img = cv2.imread(r"C:\Users\MUSA\Desktop\Nesne Tespiti\KontorAlgilama\kontor.png", 0)

# Görüntünün başarıyla yüklendiğini kontrol et
if img is None:
    print("Hata: Görüntü dosyası yüklenemedi! Lütfen dosya yolunu kontrol edin.")
    exit()

# Eşikleme (Threshold) işlemi uygula
_, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

"""
cv2.threshold(): Gri tonlamadaki görüntü üzerinde, belirli bir eşik değeri (128) kullanarak ikili (binary) görüntü
elde eder. 128'in üzerindeki piksel değerleri 255 (beyaz), altındaki piksel değerleri ise 0 (siyah) olarak atanır.
"""

# Konturları bul
contours, hierarch = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

"""
cv2.findContours(): İkili görüntü üzerinde konturları bulur. Bu fonksiyon, aynı zamanda konturların hiyerarşisini de döndüren bir sonuç verir.
cv2.RETR_CCOMP: Hem harici (dış) hem de dahili (iç) konturları bulur.
cv2.CHAIN_APPROX_SIMPLE: Kontur noktalarını sıkıştırarak sadece köşe noktalarını saklar.
"""

# Eğer kontur bulunamazsa programı durdur
if hierarch is None or len(contours) == 0:
    print("Hata: Hiç kontur bulunamadı!")
    exit()

"""
Eğer herhangi bir kontur tespit edilmezse, kullanıcıya hata mesajı gösterilir ve program sonlanır.
"""

# Harici ve dahili konturlar için boş görüntüler oluştur (uint8 formatında)
external_contour = np.zeros(img.shape, dtype=np.uint8)
internal_contour = np.zeros(img.shape, dtype=np.uint8)

"""
external_contour ve internal_contour değişkenleri, harici ve dahili konturları gösterecek şekilde sıfırlanmış (siyah) görüntülerdir.
"""

# Konturlar üzerinde dön
for i in range(len(contours)):
    if hierarch[0][i][3] == -1:  # Harici kontur
        cv2.drawContours(external_contour, contours, i, 255, -1)
    else:  # Dahili kontur
        cv2.drawContours(internal_contour, contours, i, 255, -1)

"""
hierarch[0][i][3] == -1: Bu kontrol, konturun harici olup olmadığını belirler. Eğer değer -1 ise, bu kontur harici bir konturdur.
cv2.drawContours(): Konturları belirtilen görüntü üzerine çizer. 255 ile çizilen konturlar beyaz olur.
"""        

# Harici konturları göster
plt.figure()
plt.imshow(external_contour, cmap="gray")
plt.axis("off")

# Dahili konturları göster
plt.figure()
plt.imshow(internal_contour, cmap="gray")
plt.axis("off")

plt.show()
