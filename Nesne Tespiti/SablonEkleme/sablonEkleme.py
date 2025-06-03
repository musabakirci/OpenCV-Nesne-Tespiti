import cv2
import matplotlib.pyplot as plt

# Ana görüntüyü yükle
img = cv2.imread(r"C:\Users\MUSA\Desktop\Nesne Tespiti\SablonEkleme\cat.jpeg", 0)
if img is None:
    raise ValueError("Ana görüntü yüklenemedi! Dosya yolunu kontrol et.")

# Şablon görüntüyü yükle
template = cv2.imread(r"C:\Users\MUSA\Desktop\Nesne Tespiti\SablonEkleme\cat_face.jpeg", 0)
if template is None:
    raise ValueError("Şablon görüntü yüklenemedi! Dosya yolunu kontrol et.")

print(f"Ana Görüntü Boyutu: {img.shape}")
print(f"Şablon Görüntü Boyutu: {template.shape}")

# Şablon büyükse küçült
if template.shape[0] > img.shape[0] or template.shape[1] > img.shape[1]:
    scale_percent = 50  # %50 küçültme
    width = int(template.shape[1] * scale_percent / 100)
    height = int(template.shape[0] * scale_percent / 100)
    template = cv2.resize(template, (width, height))
    print(f"Yeni Şablon Boyutu: {template.shape}")

    """
    Eğer şablon görüntüsünün boyutu ana görüntüden büyükse, şablon görüntüsü %50 oranında küçültülür.
    """

w, h = template.shape

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

"""
Şablon eşleştirme işlemi için farklı yöntemler tanımlanır. Bu yöntemler şunlardır:
cv2.TM_CCOEFF: Korelasyon katsayısı.
cv2.TM_CCOEFF_NORMED: Normalized korelasyon katsayısı.
cv2.TM_CCORR: Korelasyon.
cv2.TM_CCORR_NORMED: Normalized korelasyon.
cv2.TM_SQDIFF: Kare fark.
cv2.TM_SQDIFF_NORMED: Normalized kare fark.
"""

for method in methods:
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    """
    cv2.matchTemplate(): Şablon eşleştirme işlemi yapılır. Bu fonksiyon, her bir pikselin şablonla ne kadar uyumlu olduğunu hesaplar ve sonucu res matrisinde döndürür.
    cv2.minMaxLoc(): Elde edilen res matrisindeki en küçük ve en büyük değerleri, bunların konumlarını alır. Bu, şablonun en iyi eşleşen yerinin bulunmasını sağlar.
    """

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    """
    Eşleşme ölçütüne bağlı olarak (kare fark metodu kullanılıyorsa min_loc, diğer metotlar için max_loc), şablonun en iyi eşleştiği nokta (top_left) belirlenir.
    """

    bottom_right = (top_left[0] + w, top_left[1] + h)

    img_with_rectangle = img.copy()
    cv2.rectangle(img_with_rectangle, top_left, bottom_right, 255, 2)

    """
    Eşleşen alanı dikdörtgenle çizmek için cv2.rectangle() fonksiyonu kullanılır. top_left ve bottom_right noktaları ile dikdörtgenin köşeleri belirlenir.
    """
    plt.figure()
    plt.subplot(121)
    plt.imshow(res, cmap="gray")
    plt.title("Eşleşen Sonuç")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(img_with_rectangle, cmap="gray")
    plt.title("Tespit Edilen Sonuç")
    plt.axis("off")

    plt.suptitle(str(method))
    plt.show()

    """
    Eşleşen sonuç (res) ve tespit edilen sonuç (img_with_rectangle) yan yana görselleştirilir.
    plt.subplot() ile her iki görsel ayrı ayrı birleştirilir.
    plt.suptitle() ile her bir yöntem için başlık eklenir.
    """




"""
    Özet:
    Bu kod, bir şablon görüntüsünün ana bir görüntüdeki en uygun eşleşmesini bulmak için farklı şablon eşleştirme yöntemlerini kullanır. Her yöntemin sonucunu görsel olarak gösterir ve eşleşen
    bölgeyi dikdörtgenle işaretler. Bu tür bir işlem, nesne tespiti gibi bilgisayarla görme uygulamalarında sıklıkla kullanılır.
"""
