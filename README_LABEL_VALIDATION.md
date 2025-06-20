# Etiket Doğrulama Kılavuzu

Bu kılavuz, AI modelinizin oluşturduğu etiketlerin doğruluğunu kontrol etmek için kullanabileceğiniz yöntemleri açıklar.

## 🎯 Etiket Doğrulama Yöntemleri

### 1. **Python Script ile Görsel Kontrol**

#### Kullanım:
```bash
# Basit görüntüleme
python check_labels.py

# İnteraktif görüntüleyici
python check_labels.py --interactive

# Özel klasörler için
python check_labels.py --images outputs/predictions --labels outputs/predictions
```

#### Özellikler:
- ✅ Görüntüleri bounding box'larla birlikte gösterir
- ✅ YOLO formatındaki etiketleri pixel koordinatlarına çevirir
- ✅ İnteraktif navigasyon (Previous/Next butonları)
- ✅ Doğru/Yanlış işaretleme
- ✅ Renk kodlu sınıf gösterimi

### 2. **Web Tabanlı Validator**

#### Kullanım:
```bash
# Validator'ı başlat
python label_validator.py
```

#### Tarayıcıda Açın:
```
http://localhost:5001
```

#### Özellikler:
- 🌐 Web arayüzü ile kolay kullanım
- 📊 Gerçek zamanlı istatistikler
- ⌨️ Klavye kısayolları (← → 1 2)
- 💾 Otomatik kayıt
- 📝 Yorum ekleme

### 3. **Manuel Kontrol Araçları**

#### LabelImg:
```bash
# LabelImg kurulumu
pip install labelImg

# LabelImg başlatma
labelImg
```

#### LabelMe:
```bash
# LabelMe kurulumu
pip install labelme

# LabelMe başlatma
labelme
```

## 🔍 Kontrol Edilecek Noktalar

### 1. **Bounding Box Doğruluğu**
- [ ] Kutular doğru nesneleri çevreliyor mu?
- [ ] Kutular çok büyük/küçük mü?
- [ ] Kutular nesnelerin tamamını kapsıyor mu?

### 2. **Sınıf Doğruluğu**
- [ ] Doğru sınıf etiketleri atanmış mı?
- [ ] Yanlış sınıflandırma var mı?
- [ ] Eksik sınıflandırma var mı?

### 3. **Confidence Değerleri**
- [ ] Yüksek confidence değerleri doğru mu?
- [ ] Düşük confidence değerleri yanlış mı?
- [ ] Threshold değeri uygun mu?

### 4. **Eksik Tespitler**
- [ ] Görüntüde tespit edilmemiş nesneler var mı?
- [ ] Küçük nesneler kaçırılmış mı?
- [ ] Karmaşık sahnelerde sorun var mı?

## 📊 İstatistiksel Analiz

### Doğruluk Metrikleri:
```python
# Precision (Kesinlik)
Precision = True Positives / (True Positives + False Positives)

# Recall (Duyarlılık)
Recall = True Positives / (True Positives + False Negatives)

# F1-Score
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Kalite Kontrol Listesi:
- [ ] **Precision > 0.8** (Yüksek kesinlik)
- [ ] **Recall > 0.7** (İyi kapsama)
- [ ] **F1-Score > 0.75** (Genel kalite)
- [ ] **Confidence threshold optimize edilmiş**

## 🛠️ Pratik İpuçları

### 1. **Sistematik Kontrol**
```
1. Rastgele örnekleme yapın (tüm veriyi kontrol etmek zorunda değilsiniz)
2. Farklı koşullardaki görüntüleri kontrol edin (ışık, açı, boyut)
3. Edge case'leri özellikle inceleyin
4. Augmented görüntüleri de kontrol edin
```

### 2. **Hızlı Kontrol Teknikleri**
```
- Klavye kısayollarını kullanın (← → 1 2)
- Batch işlemler yapın
- Otomatik filtreleme kullanın
- İstatistikleri takip edin
```

### 3. **Kalite İyileştirme**
```
- Yanlış etiketleri düzeltin
- Model parametrelerini ayarlayın
- Threshold değerlerini optimize edin
- Daha fazla veri ekleyin
```

## 📈 Sonuç Raporlama

### Validation Raporu Oluşturma:
```python
# validation_results.json örneği
{
  "image1.jpg": {
    "is_correct": true,
    "comments": "Perfect detection",
    "validated_at": "2024-01-01T12:00:00"
  },
  "image2.jpg": {
    "is_correct": false,
    "comments": "Missing small objects",
    "validated_at": "2024-01-01T12:05:00"
  }
}
```

### İstatistik Raporu:
```
Toplam Görüntü: 100
Doğrulanan: 85
Doğru: 72 (84.7%)
Yanlış: 13 (15.3%)
Genel Doğruluk: 84.7%
```

## 🚀 Gelişmiş Özellikler

### 1. **Otomatik Filtreleme**
```python
# Düşük confidence'li etiketleri filtrele
python check_labels.py --min-confidence 0.5

# Belirli sınıfları göster
python check_labels.py --classes 0,1,2
```

### 2. **Batch İşlemler**
```python
# Toplu doğrulama
python batch_validate.py --input-dir outputs/predictions --output-file validation_report.csv
```

### 3. **Karşılaştırma Araçları**
```python
# İki farklı model sonuçlarını karşılaştır
python compare_models.py --model1 results1 --model2 results2
```

## 📞 Destek

Herhangi bir sorun yaşarsanız:
1. Log dosyalarını kontrol edin
2. Hata mesajlarını inceleyin
3. Gerekli kütüphanelerin kurulu olduğundan emin olun
4. Dosya yollarının doğru olduğunu kontrol edin

---

**Not:** Etiket doğrulama, model performansını artırmak için kritik bir adımdır. Düzenli kontroller yaparak modelinizi sürekli iyileştirebilirsiniz. 