import cv2
import matplotlib.pyplot as plt
import numpy as np

# resmi içe aktarma
img = cv2.imread(r"C:\Users\MUSA\Desktop\Nesne Tespiti\KoseAlgilama\sudoku.png", 0)
img = np.float32(img)
print(img.shape)
 
"""
np.float32(img): Görüntüdeki piksel değerlerinin veri tipini float32'ye dönüştürür. Bu dönüşüm, bazı algoritmalar için gereklidir (örneğin Harris köşe tespiti).
print(img.shape): Görüntünün boyutlarını (yükseklik, genişlik) yazdırır.
""" 
plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")

"""
plt.imshow(): Görüntüyü gri tonlamada gösterir.
plt.axis("off"): Eksenleri (x, y) gizler.
"""

# harris corner detection
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)

"""
cv2.cornerHarris(): Harris köşe tespiti algoritmasını uygular. Bu algoritma, görüntüdeki köşe noktalarını tespit etmeye çalışır.
blockSize = 2: Köşe hesaplaması için kullanılan bölgeyi tanımlar (burada 2x2'lik bir alan).
ksize = 3: Sobel operatörünün kernel boyutudur.
k = 0.04: Algoritmanın hassasiyetini belirleyen sabittir.
"""

plt.figure()
plt.imshow(dst, cmap = "gray")
plt.axis("off")

"""
dst: Harris köşe tespitinin çıktısıdır. Bu çıktı, köşe olma olasılığına göre parlaklık değeri yüksek olan bölgeleri gösterir.
Görselleştirilen bu görüntüde, köşe bölgeleri beyaz renkte görünür.
"""

dst = cv2.dilate(dst, None)
img[dst>0.2*dst.max()] = 1

"""
cv2.dilate(): Görüntüdeki parlak bölgeleri büyütür. Bu, tespit edilen köşelerin daha belirgin hale gelmesini sağlar.
img[dst>0.2*dst.max()] = 1: Tespit edilen köşe noktalarına beyaz (1) değeri atanır.
"""

plt.figure()
plt.imshow(dst, cmap = "gray")
plt.axis("off")

"""
Görüntüdeki köşe noktalarının daha belirgin olduğu final halini gösterir.
"""

# shi tomasi detection
img = cv2.imread(r"C:\Users\MUSA\Desktop\Nesne Tespiti\KoseAlgilama\sudoku.png", 0)
img = np.float32(img)
corners = cv2.goodFeaturesToTrack(img, 130, 0.01, 10)
corners = np.int64(corners)

"""
130: Tespit edilecek maksimum köşe sayısı.
0.01: Bu, minimum köşe tespiti kalitesini belirler (daha düşük değerler daha fazla köşe tespit eder).
10: Köşeler arasındaki mesafe için minimum sınır.
np.int64(corners): Tespit edilen köşe noktalarını tam sayı tipine dönüştürür.
"""

for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x,y), 3, (125,125,125), cv2.FILLED)
plt.imshow(img)
plt.axis("off")   

"""
for i in corners: Tespit edilen her bir köşe için döngüye girilir.
x, y = i.ravel(): Köşe noktasının x ve y koordinatları ayrılır.
cv2.circle(): Tespit edilen köşe noktalarına 3 piksel çapında beyaz bir daire çizer.
Son olarak, tüm köşe noktaları ile işaretlenmiş görüntü ekrana getirilir.
"""

plt.show()


"""
Özet:
Bu kod, iki farklı köşe tespiti algoritmasını (Harris ve Shi-Tomasi) kullanarak bir görüntüdeki köşe noktalarını tespit eder.
İlk olarak, Harris köşe tespiti ile köşe noktaları belirlenir ve büyütülerek görselleştirilir.
Ardından, Shi-Tomasi köşe tespiti kullanılarak daha fazla köşe tespiti yapılır ve her bir köşe üzerine daireler çizilir.
"""