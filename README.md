# AI Model Inference & Labeling Tool

Bu web uygulaması, derin öğrenme modellerini kullanarak görüntü analizi ve etiketleme yapmanızı sağlar. Desteklenen modeller:

- YOLOv5 ve YOLOv8 (Nesne Tespiti)
- Detectron2 (Nesne Tespiti / Segmentasyon)
- Segment Anything Model (SAM) (Segmentasyon)

## Özellikler

- Model dosyası yükleme (.pt, .pth, .pkl, .onnx)
- Toplu görsel yükleme (.zip veya .rar arşivleri)
- Otomatik model türü tespiti
- YOLO formatında etiket dosyası oluşturma
- Opsiyonel veri artırma (augmentation)
- Sonuçları zip olarak indirme

## Kurulum

1. Python 3.8+ gereklidir

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Uygulamayı başlatın:
```bash
python app.py
```

5. Tarayıcınızda http://localhost:5000 adresine gidin

## Kullanım

1. Model Yükleme
   - Desteklenen model dosyalarınızı (.pt, .pth, .pkl, .onnx) yükleyin
   - Sistem otomatik olarak model türünü tespit edecektir

2. Veri Yükleme
   - Görsellerinizi içeren bir .zip veya .rar dosyası yükleyin
   - Sistem arşivi açıp görselleri hazırlayacaktır

3. İşleme
   - "Start Processing" butonuna tıklayın
   - İsterseniz veri artırma seçeneğini aktifleştirin

4. Sonuçlar
   - İşlem tamamlandığında sonuçları indirebilirsiniz
   - Çıktı klasöründe:
     - Tahmin edilmiş görseller
     - YOLO formatında .txt etiket dosyaları
     - Veri artırma yapıldıysa ek görseller ve etiketler

## Desteklenen Formatlar

- Görseller: .jpg, .jpeg, .png
- Arşivler: .zip, .rar
- Modeller: .pt (YOLO), .pth (Detectron2/SAM), .pkl, .onnx

## Lisans

MIT License 