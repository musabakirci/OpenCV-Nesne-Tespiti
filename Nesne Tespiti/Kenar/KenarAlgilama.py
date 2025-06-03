import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
cv2: OpenCV kütüphanesini içeri aktarır, bu kütüphane görüntü işleme ve bilgisayarla görme işlemleri için kullanılır.
matplotlib.pyplot: Görüntülerin grafiksel olarak gösterilmesini sağlar.
numpy: Matematiksel işlemler için kullanılır, özellikle görüntü işleme işlemleri için önemli olan array (dizi) manipülasyonları için kullanılır.
"""

# Resmi içe aktar
img = cv2.imread(r"C:\Users\MUSA\Desktop\Nesne Tespiti\Kenar\london.jpeg", 0)

"""
Bu satır, london.jpeg adlı görüntüyü okuyup, img değişkenine atar. 0 parametresi, görüntüyü gri tonlamada (grayscale) okumasını sağlar.
"""

# Orijinal resmi göster
plt.figure()
plt.imshow(img, cmap="gray")  # cmap doğru şekilde yazıldı
plt.axis("off")

"""
plt.imshow(img, cmap="gray"): Görüntüyü gri tonlarda (gray) görselleştirir.
plt.axis("off"): Görüntü etrafındaki eksenleri kaldırır.
"""

# Kenar algılama işlemi
edges = cv2.Canny(image=img, threshold1=0, threshold2=255)

"""
Canny kenar algılama algoritması, görüntüdeki kenarları bulmak için kullanılan bir yöntemdir. Burada, threshold1=0 ve threshold2=255 ile kenar algılama için minimum ve maksimum eşik değerleri belirlenmiştir. 
Bu değerlerle kenarları tespit etmeye çalışır.
"""

# Kenar algılanmış görüntüyü göster
plt.figure()
plt.imshow(edges, cmap="gray")  # 'camp' yerine 'cmap' kullanıldı
plt.axis("off")

"""
Kenar algılama sonucu elde edilen edges görüntüsünü gri tonlama ile gösterir.
"""

med_val = np.median(img)
print(med_val)

low = int(max(0, (1 - 0.33)*med_val))
high = int(min(255, (1 + 0.33)*med_val))

"""
np.median(img): Görüntünün piksel değerlerinin medyanını hesaplar.
low ve high değişkenleri, medyan değeri kullanılarak yeni eşik değerleri hesaplanır. Bu işlem, kenar algılama için daha uygun eşik değerlerinin bulunmasına yardımcı olur.
"""

print(low)
print(high)

edges = cv2.Canny(image=img, threshold1=low, threshold2=high)

"""
Burada, daha önce hesaplanan low ve high eşik değerleri kullanılarak kenar algılama yapılır. Bu daha hassas bir kenar algılama sağlar.
"""

plt.figure()
plt.imshow(edges, cmap="gray")  # 'camp' yerine 'cmap' kullanıldı
plt.axis("off")

#blur
blurred_img = cv2.blur(img, ksize=(3,3))

"""
Görüntüyü bulanıklaştırmak için cv2.blur() fonksiyonu kullanılır. Bu, görüntüyü daha yumuşak hale getirir ve kenar algılama sırasında daha az gürültü olmasını sağlar. ksize=(3,3) parametresi, bulanıklaştırma için kullanılan çekirdek boyutunu belirtir.
"""

plt.figure()
plt.imshow(blurred_img, cmap="gray")  # 'camp' yerine 'cmap' kullanıldı
plt.axis("off")

med_val = np.median(blurred_img)
print(med_val)

low = int(max(0, (1 - 0.33)*med_val))
high = int(min(255, (1 + 0.33)*med_val))

"""
Burada, önce bulanıklaştırılmış görüntüdeki medyan değeri hesaplanır.
Ardından, medyan değeri kullanılarak uygun eşik değerleri low ve high hesaplanır ve tekrar Canny kenar algılama uygulanır.
"""

print(low)
print(high)

edges = cv2.Canny(image=blurred_img, threshold1=low, threshold2=high)

plt.figure()
plt.imshow(edges, cmap="gray")  # 'camp' yerine 'cmap' kullanıldı
plt.axis("off")



plt.show()  # Matplotlib pencerelerini görüntülemek için eklenebilir


"""
Sonuç:

İlk olarak, görüntüdeki kenarları tespit etmek için Canny algoritması kullanılır.
Ardından, median değeri hesaplanarak daha uygun eşik değerleri ile kenar algılama yapılır.
Görüntü bulanıklaştırılarak kenar algılama süreci iyileştirilir.
Son olarak, her bir adım görselleştirilerek sonuçlar kullanıcıya sunulur.
"""