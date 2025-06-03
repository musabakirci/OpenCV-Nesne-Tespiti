import cv2
import numpy as np
from collections import deque

"""
cv2: OpenCV kütüphanesi, görüntü işleme işlemlerini yapmak için kullanılır.
numpy: Sayısal işlemler ve dizi işlemleri için kullanılır.
deque: Nesnenin hareketini takip etmek için çift uçlu bir liste (deque) kullanılır.
"""

buffer_size = 16
pts = deque(maxlen=buffer_size)

"""
buffer_size = 16: Takip edilen nesnenin 16 çerçeveye kadar hareketinin kaydedilmesini sağlar.
pts = deque(maxlen=buffer_size): Takip edilen noktaları bir deque (çift uçlu liste) içinde saklar.
"""

# Mavi renk aralığı (HSV)
blueLower = (84, 98, 0)
blueUpper = (179, 255, 255)

"""
HSV formatında mavi renk aralığını belirler. HSV (Hue, Saturation, Value), renkleri daha iyi algılamamızı sağlar.
blueLower: Mavi rengin en düşük değeri.
blueUpper: Mavi rengin en yüksek değeri.
"""

# Kamera başlat
cap = cv2.VideoCapture(0)
cap.set(3, 960)  # Genişlik
cap.set(4, 480)  # Yükseklik

"""
cap = cv2.VideoCapture(0): Bilgisayarın varsayılan kamerasını başlatır.
cap.set(3, 960): Kameranın genişliğini 960 piksel olarak ayarlar.
cap.set(4, 480): Kameranın yüksekliğini 480 piksel olarak ayarlar.
"""

while True:
    success, imgOriginal = cap.read()

    if not success:
        print("Hata: Kameradan görüntü alinamadi!")
        break

    """
    cap.read(): Kameradan görüntü okur.
    Eğer görüntü alınamazsa, hata mesajı verir ve döngüden çıkar.
    """

    # Gaussian Blur
    blurred = cv2.GaussianBlur(imgOriginal, (11, 11), 0)

    """
    Gaussian Blur filtresi, görüntüdeki gürültüyü azaltır ve kenarları daha belirgin hale getirir.
    """

    # BGR'den HSV'ye dönüştür
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    """
    OpenCV, varsayılan olarak BGR formatında çalışır.
    cv2.COLOR_BGR2HSV: BGR'den HSV'ye dönüştürür, çünkü renk algılama HSV formatında daha başarılıdır.
    """

    # HSV görüntüsünü göster
    cv2.imshow("HSV Image", hsv)

    """
    cv2.inRange(): HSV formatında belirlenen mavi renk aralığındaki pikselleri beyaz (255), diğer pikselleri siyah (0) yapar.
    """

    # Mavi için maske oluştur
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    cv2.imshow("Mask Image", mask)

    # Maskenin etrafında kalan gürültüleri sil
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow("Mask + Erozyon ve Genisleme", mask)

    """
    Erozyon (erode): Küçük gürültüleri temizler.
    Genişleme (dilate): Erozyon sonrası kaybolan önemli bilgileri geri getirir.
    """

    # Konturları bul
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    """
    cv2.findContours(): Görüntüdeki nesne konturlarını (sınırlarını) bulur.
    cv2.RETR_EXTERNAL: Sadece dış konturları alır.
    cv2.CHAIN_APPROX_SIMPLE: Kontur noktalarını optimize eder, gereksiz noktaları kaldırır.
    """

    center = None

    if len(contours) > 0:

        # En büyük konturu al
        c = max(contours, key=cv2.contourArea)

        """
        Eğer kontur bulunduysa, en büyük alanı kaplayan kontur seçilir.
        """

        # Dikdörtgene çevir
        rect = cv2.minAreaRect(c)

        ((x, y), (width, height), rotation) = rect

        """
        cv2.minAreaRect(): Konturun etrafına dönebilen bir dikdörtgen çizer.
        x, y: Nesnenin merkez noktası.
        width, height: Dikdörtgenin genişliği ve yüksekliği.
        rotation: Dikdörtgenin dönüş açısı.
        """

        # Koordinatları yazdır
        s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x), np.round(y), np.round(width), np.round(height), np.round(rotation))
        print(s)

        """
        Koordinatları ve boyutları ekrana yazdırır.
        """

        # Konturun etrafına dikdörtgen çiz
        box = cv2.boxPoints(rect)
        box = np.int64(box)  # np.int0 yerine np.int32 kullanılıyor

        """
        cv2.boxPoints(rect): Dikdörtgenin köşe noktalarını bulur.
        """

        # moment
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

        """
        Moment hesaplamaları ile nesnenin ağırlık merkezini bulur.
        """

        # konturu çizdir: sarı
        cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)

        """
        cv2.drawContours(): Sarı ((0,255,255)) renkte dikdörtgen çizer.
        """


        # Merkeze bir tane nokta çizelim: pembe
        cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)

        """
        cv2.circle(): Nesnenin merkezine pembe ((255,0,255)) bir nokta çizer.
        """

        # Çizgileri ekrana yazdir
        cv2.putText(imgOriginal, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

        """
        cv2.putText() fonksiyonu, görüntü üzerine yazı eklemek için kullanılır.
        s: Yazdırılacak metin (x, y, genişlik, yükseklik ve dönüş açısı bilgileri).
        (25, 50): Metnin görüntü üzerinde konumlandığı koordinatlar (sol üst köşeye yakın).
        cv2.FONT_HERSHEY_COMPLEX_SMALL: Yazı tipi.
        1: Yazı boyutu.
        (255, 255, 255): Beyaz renk (BGR formatında).
        2: Kalınlık (stroke).
        """

    # deque(takip)
    pts.appendleft(center)

    for i in range(1, len(pts)):

        if pts[i -1] is None or pts[i] is None: continue

        cv2.line(imgOriginal, pts[i -1], pts[i], (0, 255, 0), 3)

        """
        pts.appendleft(center): Merkez koordinatlarını deque içine ekler.
        cv2.line(): Önceki ve yeni merkez noktalarına yeşil ((0,255,0)) çizgiler çizerek hareket yolunu gösterir.
        """

        # Sonuçları göster
        cv2.imshow("Orijinal Tespit", imgOriginal)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()

"""
'q' tuşuna basıldığında programdan çıkılır.
Kamera serbest bırakılır ve tüm OpenCV pencereleri kapatılır.
"""
