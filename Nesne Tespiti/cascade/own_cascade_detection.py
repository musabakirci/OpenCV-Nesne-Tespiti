import cv2

path = r"C:\Users\MUSA\Desktop\Nesne Tespiti\cascade\cascade.xml"
objectName = "Mouse"
frameWidth = 280
frameHeight = 360
color = (255,0,0)

"""
path: Cascade siniflandiricisi dosya yolu.
objectName: Algilanan nesnenin ekrana yazilacak adi.
frameWidth, frameHeight: Kamera goruntusunun genislik ve yukseklik degerleri.
color: Dikdortgen cizimi icin renk (BGR formatinda).
"""

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

"""
Kamera baslatiliyor.
Cozunurluk ayarlaniyor.
"""

# trackbar
def empty(a): 
    pass

"""
Bos bir fonksiyon tanimlaniyor.
Burada, trackbar degisikliklerini takip eden bos bir fonksiyon olarak tanimlaniyoruz.
"""
       

cv2.namedWindow("Sonuc")
cv2.resizeWindow("Sonuc", frameWidth, frameHeight + 100)
cv2.createTrackbar("Scale","Sonuc",400,1000,empty)
cv2.createTrackbar("Neighbor","Sonuc",4,50, empty)

"""
Trackbarlar ekleniyor.
Scale: Eklenen trackbar, resim boyutunun %50 oranında küçültülmesini saglar.
Neighbor: Eklenen trackbar, resimdeki nesnelerin arasında bulunan maksimum uzaklık degerini belirler.
"""

# cascade classifier
cascade = cv2.CascadeClassifier(r"C:\Users\MUSA\Desktop\Nesne Tespiti\cascade\cascade.xml")


while True:

    """
    Sonsuz dongu baslatiliyor.
    """

    # read img
    success, img = cap.read()

    """
    Kameradan goruntu aliniyor.
    """

    if success:

        # convert bgr2gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detection parameters
        scaleVal = 1 + (cv2.getTrackbarPos("Scale","Sonuc")/1000)
        neighbor = cv2.getTrackbarPos("Neighbor", "Sonuc")

        """
        Kullanici tarafindan ayarlanan degerler aliniyor.
        """

        # detection 
        rects = cascade.detectMultiScale(gray, scaleVal, neighbor)

        """
        Nesne tespiti yapiliyor.
        """

        for(x,y,w,h) in rects:

            cv2.rectangle(img, (x,y), (x+w, y+h), color, 3)
            cv2.putText(img, objectName, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            """
            Dikdortgen ciziliyor.
            Nesne adi ekrana yazdiriliyor.
            """


        cv2.imshow("Sonuc", img) 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()   
cv2.destroyAllWindows()


"""
Nesne Tespit Algoritmasi
Kutuphaneleri Dahil Et

OpenCV'yi iceri aktar.
Degiskenleri Tanimla

Cascade dosya yolunu belirle.
Nesne adini tanimla.
Kamera cerceve boyutlarini ayarla.
Dikdortgen cizimi icin renk belirle.
Kamerayi Baslat ve Cozunurlugu Ayarla

Kameradan goruntu almak icin cv2.VideoCapture(0) kullan.
cap.set() fonksiyonu ile genislik ve yuksekligi belirle.
Trackbar Olustur

Trackbar degisikliklerini takip eden bos bir fonksiyon tanimla.
cv2.namedWindow() ile pencere olustur.
cv2.createTrackbar() ile Scale ve Neighbor trackbarlarini tanimla.
Cascade Siniflandiriciyi Yukle

cv2.CascadeClassifier(path) ile cascade modelini yukle.
Model yuklenemezse hata mesaji yazdir ve programi durdur.
Donguyu Baslat

Kameradan surekli goruntu al.
Alinan goruntuyu gri tonlamaya cevir.
Trackbar degerlerine gore scaleFactor ve minNeighbors parametrelerini al.
detectMultiScale() fonksiyonu ile nesne tespiti yap.
Tespit edilen nesnelerin etrafina dikdortgen ciz ve nesne adini ekrana yazdir.
Sonucu Goster

cv2.imshow() ile islenmis goruntuyu goster.
Cikis Yap

Kullanici "q" tusuna bastiginda donguyu durdur.
"""






