import cv2
import matplotlib.pyplot as plt
import numpy as np


# Görseli içe aktar
coin = cv2.imread(r"C:\Users\MUSA\Desktop\Nesne Tespiti\HavzaAlgoritmasi\foto.jpg") 

# BGR'den RGB'ye dönüştürme
coin_rgb = cv2.cvtColor(coin, cv2.COLOR_BGR2RGB)

# Görseli görselleştirme
plt.figure()
plt.imshow(coin_rgb)
plt.axis("off")
plt.title("Orijinal Görsel")

# LPF (Low Pass Filter) - Blur işlemi
coin_blur = cv2.medianBlur(coin, 13)
coin_blur_rgb = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2RGB)

# Blur sonrası  
plt.figure()
plt.imshow(coin_blur_rgb)
plt.axis("off")
plt.title("Blur Uygulandı")

# Grayscale dönüşümü
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)

# Grayscale görseli gösterme
plt.figure()
plt.imshow(coin_gray, cmap="gray")
plt.axis("off")
plt.title("Grayscale Görsel")

# Thresholding işlemi
ret, coin_thresh = cv2.threshold(coin_gray, 75, 255, cv2.THRESH_BINARY)

# Threshold sonrası görsel
plt.figure()
plt.imshow(coin_thresh, cmap="gray")
plt.axis("off")
plt.title("Threshold Sonucu")

# kontur

contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):

    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coin, contours, i, (0,255,0),10)

# Açılma (Morfolojik işlem)
kernel = np.ones((5, 5), np.uint8)  # Kernel boyutunu 3x3'ten 5x5'e yükselttik
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Açılma sonrası görsel
plt.figure()
plt.imshow(opening, cmap="gray")
plt.axis("off")
plt.title("Açılma Sonucu")        

plt.figure()
plt.imshow(coin)
plt.axis("off")

# nesneler arası distance bulalım
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
plt.figure()
plt.imshow(opening, cmap="gray")
plt.axis("off")

# resmi küçült
ret, sure_foreground = cv2.threshold(dist_transform, 0.4*np.max(dist_transform), 255, 0)
plt.figure()
plt.imshow(dist_transform, cmap="gray")
plt.axis("off")

# arka plan için resmi büyüt
sure_background = cv2.dilate(opening, kernel, iterations = 1)
sure_foreground = np.uint8(sure_foreground)
unknown = cv2.subtract(sure_background, sure_foreground)
plt.figure()
plt.imshow(unknown, cmap = "gray")
plt.axis("off")

# bağlantı
ret, marker = cv2.connectedComponents(sure_foreground)
marker = marker + 1
marker[unknown == 255] = 0
plt.figure()
plt.imshow(marker, cmap="gray")
plt.axis("off")

# havza
marker = cv2.watershed(coin, marker)
plt.figure()
plt.imshow(marker, cmap="off")
plt.axis("off")

contours, hierarchy = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


for i in range(len(contours)):

    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coin, contours, i, (255,0,0),10)

plt.figure()
plt.imshow(coin)
plt.axis("off")

plt.show()







