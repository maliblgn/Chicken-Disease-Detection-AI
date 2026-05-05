---
title: CİK-SÖR Akustik Erken Uyarı Sistemi
emoji: 🐔
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

# CİK-SÖR V1 Demo

## Proje Amacı

CİK-SÖR, kanatlı hayvanlarda özellikle tavukların vokalizasyonlarını analiz ederek sağlıklı, anormal/hastalık şüphesi taşıyan ve çevresel gürültü sınıflarını ayırmayı amaçlayan bir yapay zekâ destekli erken uyarı sistemidir.

> **Önemli:** Bu sistem kesin teşhis aracı değildir. Erken uyarı ve anomali tespiti amaçlıdır. Nihai model, saha donanımıyla toplanacak veteriner onaylı verilerle yeniden eğitilecektir.

## Sistem Akışı

```text
Ses dosyası yükle veya mikrofondan kısa kayıt al
↓
Sesi normalize et / mono hale getir / örnekleme hızını standartlaştır (22050 Hz)
↓
Gürültü azaltma uygula (Spektral Çıkarma)
↓
Sesi 5 saniyelik segmentlere böl
↓
Spektrogram / MFCC / Delta / Delta-Delta özellikleri çıkar
↓
Model tahmini yap (RF + CNN)
↓
Healthy / Unhealthy / Noise sonucunu güven skoru ile göster
↓
Anormal sonuç varsa uyarı üret
↓
Sonucu SQLite'a logla
```

## Kurulum

```bash
# Sanal ortam oluştur ve aktif et
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Bağımlılıkları yükle
pip install -r requirements.txt
```

## Çalıştırma

```bash
python app.py
```

Tarayıcınızda `http://127.0.0.1:7860` adresini açın.

## Veri Klasör Yapısı

```text
Kayıtlar/           → Orijinal etiketli ses kayıtları (Healthy, Unhealthy, Noise)
Veri_Egitim/         → Çoğaltılmış eğitim verisi
Veri_Test/           → Test verisi
Spektrogramlar/      → CNN için 3-kanallı RGB görseller
data/logs/           → SQLite analiz logları
```

## Model Dosyaları

Eğitilmiş modeller `models/` klasöründe bulunur:

| Dosya | Açıklama |
|-------|----------|
| `tavuk_modeli.pkl` | Random Forest (120 boyut MFCC+Delta) |
| `tavuk_cnn_modeli.keras` | 3-Kanallı CSCNN (128x128x3 RGB) |
| `tavuk_edge_model.tflite` | TFLite Edge AI (opsiyonel) |

Model dosyası yoksa sistem demo/mock mode ile çalışır.

## Demo Kullanımı

1. `python app.py` ile paneli açın.
2. **Sistem Durumu** sekmesinden modellerin yüklü olduğunu kontrol edin.
3. **Dosya ile Test** sekmesinden bir ses dosyası yükleyin.
4. Sonucu, güven skorunu ve spektrogramı inceleyin.
5. **Loglar** sekmesinden analiz geçmişini görüntüleyin.

## Canlı Mikrofon Analizi

**Canlı Analiz** sekmesinden 5 saniyelik kayıt alabilirsiniz. Mikrofon erişimi gerektirir.

## Sınıflar

| Teknik Sınıf | Arayüz Karşılığı |
|--------------|-------------------|
| Healthy | Normal Vokalizasyon |
| Unhealthy | Anormal / Hastalık Şüphesi Taşıyan Vokalizasyon |
| Noise | Gürültü / Analiz Dışı Ses |

## Bilimsel Not

Bu demo modeli, sistem mimarisinin test edilmesi amacıyla hazırlanmıştır. Mevcut veri seti internetten toplanmış ve çoğaltma teknikleriyle genişletilmiş seslerden oluşmaktadır. Nihai model, saha donanımıyla toplanacak veteriner onaylı verilerle yeniden eğitilecektir.

## Sınırlılıklar

- Kesin hastalık teşhisi iddia edilmemektedir.
- Sahadan toplanmamış veriyle yüksek doğruluk iddiası yapılmamaktadır.
- Bu sürümde çevresel sensörler (sıcaklık, nem, CO₂, NH₃) bulunmamaktadır.
- Sistem yalnızca akustik veri üzerinden çalışmaktadır.

## Gelecek Geliştirmeler

- Gerçek sağlıklı/hasta tavuk seslerinin saha donanımıyla toplanması
- Veteriner gözlemli anormal seslerin etiketlenmesi
- Jetson Nano üzerinde edge inference testi
- 7/24 ring buffer ile sürekli dinleme sistemi
- Bildirim sistemi (web/mobil)
- Çevresel sensör entegrasyonu
