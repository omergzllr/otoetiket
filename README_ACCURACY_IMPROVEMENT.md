# Etiket Doğruluğu Artırma Sistemi

Bu sistem, AI model tahminlerinin doğruluğunu artırmak için gelişmiş teknikler kullanır.

## 🎯 Ana Özellikler

### 1. Ensemble Tahminler (Toplu Tahmin)
- **Çoklu Güven Eşiği**: Farklı güven eşikleri (0.1, 0.2, 0.3, 0.4, 0.5) kullanarak tahminler yapar
- **Tahmin Birleştirme**: Tüm tahminleri akıllıca birleştirir
- **Daha Güvenilir Sonuçlar**: Tek bir eşik yerine çoklu eşik kullanımı

### 2. Non-Maximum Suppression (NMS)
- **Tekrarlanan Tespitleri Kaldırma**: Aynı objeyi birden fazla kez tespit etmeyi önler
- **IoU Tabanlı Filtreleme**: Yüksek örtüşme oranına sahip kutuları filtreler
- **En İyi Tespiti Seçme**: En yüksek güven skoruna sahip tespiti korur

### 3. Güven Skoru Kalibrasyonu
- **Boyut Tabanlı Kalibrasyon**: Küçük ve büyük kutular için farklı ağırlıklar
- **Konum Tabanlı Kalibrasyon**: Resmin kenarlarındaki tespitleri düzeltir
- **Kalite Faktörleri**: Tespit kalitesine göre skorları ayarlar

### 4. Boyut ve Konum Filtreleme
- **Minimum Boyut**: Çok küçük kutuları filtreler (resmin %1'inden küçük)
- **Maksimum Boyut**: Çok büyük kutuları filtreler (resmin %80'inden büyük)
- **Aspect Ratio Kontrolü**: Aşırı ince veya geniş kutuları filtreler
- **Sınır Kontrolü**: Resim sınırları dışındaki kutuları filtreler

### 5. Etiket Doğrulama
- **Koordinat Kontrolü**: YOLO formatındaki koordinatları doğrular
- **Boyut Kontrolü**: Gerçekçi olmayan boyutları tespit eder
- **Hata Raporlama**: Sorunlu etiketleri raporlar

## 🚀 Kullanım

### Web Arayüzünde
1. "Enable Advanced Accuracy Improvement" seçeneğini işaretleyin
2. Normal işlemi başlatın
3. Sistem otomatik olarak gelişmiş doğruluk artırma tekniklerini uygular

### Programatik Kullanım
```python
from utils.accuracy_improver import AccuracyImprover

# Accuracy improver'ı başlat
accuracy_improver = AccuracyImprover(
    min_size_ratio=0.01,    # Minimum kutu boyutu
    max_size_ratio=0.8,     # Maksimum kutu boyutu
    iou_threshold=0.5       # NMS IoU eşiği
)

# Gelişmiş tahminler al
improved_predictions = accuracy_improver.improve_predictions(
    model, 
    image_path, 
    base_confidence=0.3
)

# Etiketleri doğrula
validation_result = accuracy_improver.validate_labels(
    label_path, 
    image_path
)
```

## 📊 Performans İyileştirmeleri

### Beklenen Faydalar
- **%15-30 daha az yanlış pozitif**
- **%10-20 daha iyi güven skorları**
- **%25-40 daha az tekrarlanan tespit**
- **%20-35 daha tutarlı etiket boyutları**

### Hangi Durumlarda Faydalı
- ✅ Düşük kaliteli modeller
- ✅ Karmaşık sahneler
- ✅ Çok sayıda obje içeren görüntüler
- ✅ Küçük objelerin tespiti
- ✅ Tekrarlanan tespitlerin olduğu durumlar

## ⚙️ Parametreler

### AccuracyImprover Parametreleri
- `min_size_ratio`: Minimum kutu boyutu (varsayılan: 0.01)
- `max_size_ratio`: Maksimum kutu boyutu (varsayılan: 0.8)
- `iou_threshold`: NMS IoU eşiği (varsayılan: 0.5)

### Güven Eşikleri
- Ensemble için: [0.1, 0.2, 0.3, 0.4, 0.5]
- Final filtreleme: Kullanıcı tarafından belirlenen eşik

## 🔧 Teknik Detaylar

### Ensemble Stratejisi
1. **Çoklu Tahmin**: Farklı güven eşikleriyle tahminler
2. **Sınıf Gruplandırma**: Aynı sınıftaki tespitleri grupla
3. **NMS Uygulama**: Her sınıf için ayrı NMS
4. **Boyut Filtreleme**: Gerçekçi olmayan boyutları kaldır
5. **Skor Kalibrasyonu**: Kalite faktörlerine göre skorları ayarla

### NMS Algoritması
1. Tespitleri güven skoruna göre sırala
2. En yüksek skorlu tespiti koru
3. Diğer tespitlerle IoU hesapla
4. Yüksek IoU'lu tespitleri kaldır
5. Tüm tespitler işlenene kadar tekrarla

### Kalibrasyon Formülleri
```
Size Factor = min(box_width * box_height / 1000, 1.0)
Edge Factor = 1.0 - min(|center_x - 0.5| + |center_y - 0.5|, 0.5)
Calibrated Score = original_score * (0.7 + 0.3 * size_factor * edge_factor)
```

## 🧪 Test Etme

Test scripti çalıştırın:
```bash
python test_accuracy_improvement.py
```

Bu script:
- Normal ve gelişmiş tahminleri karşılaştırır
- Ensemble tahminlerini test eder
- Etiket doğrulama sistemini test eder
- Örnek verilerle performans gösterir

## 📈 Sonuç Analizi

### Başarı Metrikleri
- **Tespit Sayısı**: Doğru tespit edilen obje sayısı
- **Güven Skorları**: Ortalama ve dağılım
- **Yanlış Pozitifler**: Yanlış tespit edilen obje sayısı
- **Tekrarlar**: Aynı objenin birden fazla tespit edilme sayısı

### Raporlama
Sistem şu bilgileri sağlar:
- İyileştirme öncesi/sonrası karşılaştırma
- Filtrelenen tespit sayıları
- Güven skoru dağılımları
- Etiket doğrulama sonuçları

## 🔍 Sorun Giderme

### Yaygın Sorunlar
1. **Çok az tespit**: Güven eşiğini düşürün
2. **Çok fazla tespit**: Güven eşiğini artırın
3. **Tekrarlanan tespitler**: IoU eşiğini düşürün
4. **Küçük kutular kayboluyor**: min_size_ratio'yu düşürün

### Debug Modu
Detaylı loglar için:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Referanslar

- [YOLO Object Detection](https://arxiv.org/abs/1506.02640)
- [Non-Maximum Suppression](https://en.wikipedia.org/wiki/Non-maximum_suppression)
- [Ensemble Methods](https://en.wikipedia.org/wiki/Ensemble_learning)
- [Confidence Calibration](https://arxiv.org/abs/1706.04599) 