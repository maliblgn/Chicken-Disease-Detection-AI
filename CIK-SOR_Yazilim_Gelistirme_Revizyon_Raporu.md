# CİK-SÖR Yazılım Geliştirme ve Revizyon Raporu

## 1. Proje Bağlamı

Proje adı: **CİK-SÖR**

CİK-SÖR, kanatlı hayvanlarda özellikle tavukların vokalizasyonlarını analiz ederek sağlıklı, anormal/hastalık şüphesi taşıyan ve çevresel gürültü sınıflarını ayırmayı amaçlayan bir yapay zekâ destekli erken uyarı sistemidir.

Projenin nihai hedefi, kümes içine yerleştirilecek donanım ile tavuk seslerini sürekli dinlemek, bu sesleri işlemek, veri seti oluşturmak, derin öğrenme modeliyle analiz etmek ve anormal/hastalık şüphesi taşıyan seslerde kullanıcıya anlık bildirim vermektir.

Bu aşamadaki hedef, henüz saha materyalleri tamamen gelmeden mevcut yazılım prototipini **gerçekten çalışan, sunumda gösterilebilir, hatalara dayanıklı ve geliştirilebilir** bir hale getirmektir.

Önemli kapsam notu:  
Bu ilk sürümde sıcaklık, nem, CO₂, NH₃ gibi çevresel sensörlere odaklanılmayacaktır. Sistem yalnızca **akustik veri** üzerinden ilerleyecektir.

---

## 2. MVP Hedefi

İlk çalışır sürüm şu işi güvenilir şekilde yapmalıdır:

```text
Ses dosyası yükle veya mikrofondan kısa kayıt al
↓
Sesi normalize et / mono hale getir / örnekleme hızını standartlaştır
↓
Gürültü azaltma veya gürültü ayrımı uygula
↓
Sesi segmentlere böl
↓
Spektrogram / MFCC / Delta / Delta-Delta özellikleri çıkar
↓
Model tahmini yap
↓
Healthy / Unhealthy / Noise sonucunu güven skoru ile göster
↓
Anormal sonuç varsa uyarı üret
↓
Sonucu logla
```

Bu sürümün sunum dili şu şekilde olmalıdır:

> “CİK-SÖR V1 Demo, tavuk seslerini analiz ederek normal vokalizasyon, anormal/hastalık şüphesi taşıyan vokalizasyon ve çevresel gürültüyü ayıran çalışan bir yapay zekâ prototipidir.”

Kesin teşhis iddiası kullanılmamalıdır. “Hasta” yerine tercihen:

```text
Anormal vokalizasyon / hastalık şüphesi
```

ifadesi kullanılmalıdır.

---

## 3. Mevcut Sistemden Beklenen Temel Modüller

Mevcut projede veya revizyon sonunda şu modüler yapı hedeflenmelidir:

```text
cik-sor/
│
├── app.py                         # Ana Gradio dashboard
├── config.py                      # Sabitler, yollar, eşikler
├── requirements.txt               # Bağımlılıklar
├── README.md                      # Kurulum ve kullanım dokümanı
│
├── data/
│   ├── raw/                       # Ham sesler
│   ├── processed/                 # İşlenmiş sesler
│   ├── demo/
│   │   ├── healthy/
│   │   ├── unhealthy/
│   │   └── noise/
│   └── logs/
│
├── models/
│   ├── tavuk_modeli.pkl           # RF veya klasik model
│   ├── tavuk_cnn_modeli.keras     # CNN modeli
│   └── tavuk_edge_model.tflite    # Edge modeli, varsa
│
├── src/
│   ├── audio_io.py                # Ses okuma, kaydetme, mikrofon kaydı
│   ├── preprocessing.py           # Normalize, mono, resample, trim
│   ├── noise_filter.py            # Gürültü azaltma
│   ├── segmentation.py            # Ses segmentleme
│   ├── features.py                # MFCC, Mel, Delta, Delta-Delta
│   ├── spectrogram.py             # 128x128x3 spektrogram üretimi
│   ├── predictor.py               # Model yükleme ve tahmin
│   ├── decision.py                # Karar katmanı, eşikler
│   ├── database.py                # SQLite loglama
│   └── utils.py                   # Yardımcı fonksiyonlar
│
└── tests/
    ├── test_audio_io.py
    ├── test_preprocessing.py
    ├── test_spectrogram.py
    └── test_predictor.py
```

Ana hedef, tüm kodun tek bir dosyaya yığılmaması; arayüz, model, ses işleme ve loglama bölümlerinin ayrılmasıdır.

---

## 4. Sınıflandırma Etiketleri

İlk sürümde üç sınıf kullanılacaktır:

```text
Healthy
Unhealthy
Noise
```

Arayüzde daha kullanıcı dostu karşılıklar gösterilebilir:

```text
Healthy   → Normal vokalizasyon
Unhealthy → Anormal / hastalık şüphesi taşıyan vokalizasyon
Noise     → Gürültü / analiz dışı ses
```

Noise sınıfı kritik önemdedir. Kümes ortamında fan, yemleme, metal çarpması, insan sesi, cihaz titreşimi, su hattı ve diğer mekanik sesler modele karışabilir. Bu nedenle Noise yalnızca “çöp veri” değil, modelin aktif olarak ayırması gereken bir sınıftır.

---

## 5. Ses İşleme Standardı

Tüm sesler modele girmeden önce standart hale getirilmelidir.

Önerilen standartlar:

```text
Format: WAV
Kanal: Mono
Örnekleme hızı: 16000 Hz veya 44100 Hz
Segment uzunluğu: 5 saniye
Minimum RMS eşiği: config üzerinden ayarlanabilir
Spektrogram boyutu: 128x128x3
```

Ses okuma sırasında şu durumlar yakalanmalıdır:

- Dosya bozuksa sistem çökmesin.
- MP3, WAV, M4A gibi farklı formatlar mümkünse dönüştürülsün.
- Stereo dosya gelirse mono hale getirilsin.
- Çok kısa ses gelirse padding yapılsın veya kullanıcıya uyarı verilsin.
- Çok uzun ses gelirse segmentlere bölünsün.
- Sessiz segmentler analiz dışı bırakılabilsin.

---

## 6. Özellik Çıkarımı

Mevcut sistemde kullanılan ana fikir korunmalıdır:

```text
Mel-Spektrogram
MFCC
Delta
Delta-Delta
```

CNN girdisi için 3 kanallı yapı önerilir:

```text
R kanalı: Mel-Spektrogram
G kanalı: Delta
B kanalı: Delta-Delta
```

Bu yapı bozulmamalıdır. Model 128x128x3 bekliyorsa her zaman aynı boyutta çıktı üretilmelidir.

Ayrıca klasik model için şu özellik çıkarımı korunabilir:

```text
20 MFCC
20 Delta
20 Delta-Delta
Her biri için mean ve std
Toplam: 120 boyutlu özellik vektörü
```

---

## 7. Model Katmanı

Sistem model dosyaları bulunmadığında çökmemelidir.

Beklenen davranış:

- CNN modeli varsa yükle.
- RF modeli varsa yükle.
- TFLite modeli varsa opsiyonel olarak yükle.
- Hiç model yoksa “demo/mock mode” ile çalış ama bunu arayüzde açıkça belirt.
- Model tahmini başarısız olursa kullanıcıya anlaşılır hata ver.
- Güven skoru düşükse “kararsız tahmin” olarak göster.

Önerilen karar mantığı:

```text
confidence >= 0.75 → güçlü tahmin
0.50 <= confidence < 0.75 → düşük güvenli tahmin
confidence < 0.50 → belirsiz / tekrar analiz gerekli
```

Unhealthy sonucu sadece güven skoru belirli eşiğin üzerindeyse alarm üretmelidir.

---

## 8. Arayüz Hedefi

Ana arayüz Gradio ile hazırlanabilir. Arayüz sunumda gösterilebilir ve anlaşılır olmalıdır.

Önerilen sekmeler:

### 1. Canlı Analiz

- Mikrofondan 5 saniyelik kayıt alır.
- Sesi analiz eder.
- Sonucu gösterir.

### 2. Dosya ile Test

- Kullanıcı WAV/MP3/MP4 dosyası yükler.
- Sistem dosyayı işler.
- Sonucu ve güven skorunu gösterir.

### 3. Sinyal Görselleştirme

- Ham ses dalgası
- Temizlenmiş ses dalgası
- Spektrogram
- Tahmin sonucu

### 4. Loglar

- Tarih/saat
- Dosya adı veya kayıt tipi
- Tahmin sınıfı
- Güven skoru
- Uyarı durumu

### 5. Sistem Durumu

- Model yüklü mü?
- Mikrofon erişilebilir mi?
- Veritabanı çalışıyor mu?
- Kayıt klasörleri var mı?
- Son analiz zamanı
- Demo/mock mode aktif mi?

---

## 9. Sunum Demo Senaryosu

Sunum sırasında sistem şu akışı sorunsuz yapmalıdır:

1. Panel açılır.
2. Sistem durumu “Hazır” görünür.
3. Önce bir Noise dosyası yüklenir.
4. Sistem “Gürültü / analiz dışı ses” sonucu verir.
5. Sonra Healthy örneği yüklenir.
6. Sistem “Normal vokalizasyon” sonucu verir.
7. Sonra Unhealthy/anormal örnek yüklenir.
8. Sistem “Anormal vokalizasyon tespit edildi” uyarısı verir.
9. Log tablosunda analiz sonuçları görünür.
10. Mümkünse mikrofondan 5 saniyelik canlı kayıt alınır.

Demo günü internet bağlantısına bağımlı olunmamalıdır.

---

## 10. Hata Dayanıklılığı

Kod şu hatalara karşı dayanıklı olmalıdır:

- Model dosyası yok.
- Mikrofon bağlı değil.
- Ses dosyası bozuk.
- Ses çok kısa.
- Ses çok düşük enerjili.
- Klasörler eksik.
- SQLite dosyası yok.
- GPU yok.
- TensorFlow yüklenmemiş.
- TFLite runtime yok.
- MP4 dosyasında ses yok.
- Dosya formatı desteklenmiyor.

Her hata kullanıcıya okunabilir mesajla dönmelidir. Terminalde ayrıntılı log tutulmalıdır.

---

## 11. Loglama ve Veritabanı

SQLite kullanılabilir.

Önerilen tablo:

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source_type TEXT,
    file_name TEXT,
    predicted_class TEXT NOT NULL,
    confidence REAL,
    alert INTEGER,
    notes TEXT
);
```

Sadece Unhealthy değil, tüm analizler loglanabilir. Ancak arayüzde “uyarı logları” ayrı filtrelenebilir.

---

## 12. Demo Veri Klasörü

Materyaller gelmeden önce küçük bir demo veri klasörü hazırlanmalıdır:

```text
data/demo/
├── healthy/
│   ├── healthy_01.wav
│   └── healthy_02.wav
├── unhealthy/
│   ├── unhealthy_01.wav
│   └── unhealthy_02.wav
└── noise/
    ├── fan_noise_01.wav
    └── metal_noise_01.wav
```

Gerçek hasta verisi yoksa yapay veya temsili veri kullanılıyorsa arayüzde veya README’de bu açıkça belirtilmelidir:

> “Bu demo modeli, sistem mimarisinin test edilmesi amacıyla hazırlanmıştır. Nihai model, saha donanımıyla toplanacak veteriner onaylı verilerle yeniden eğitilecektir.”

---

## 13. Kod Kalitesi Beklentileri

Claude’den beklenen geliştirme yaklaşımı:

1. Önce mevcut repoyu analiz et.
2. Dosya yapısını çıkar.
3. Çalışmayan importları düzelt.
4. Gereksiz tekrarları temizle.
5. Kodları modüllere ayır.
6. `config.py` üzerinden ayarlanabilir yapı kur.
7. Eksik klasörleri otomatik oluşturan başlangıç fonksiyonu ekle.
8. Model dosyası yoksa demo/mock mode ekle.
9. Arayüzü sade, stabil ve sunuma uygun hale getir.
10. README dosyası oluştur.
11. `requirements.txt` dosyasını güncelle.
12. Basit testler ekle.
13. Tek komutla çalıştırılabilir hale getir.

Beklenen çalıştırma:

```bash
python app.py
```

veya

```bash
python -m app
```

---

## 14. README İçeriği

README içinde şu başlıklar bulunmalıdır:

```text
# CİK-SÖR V1 Demo

## Proje Amacı
## Sistem Akışı
## Kurulum
## Çalıştırma
## Veri Klasör Yapısı
## Model Dosyaları
## Demo Kullanımı
## Canlı Mikrofon Analizi
## Sınıflar
## Bilimsel Not
## Sınırlılıklar
## Gelecek Geliştirmeler
```

README’de kesin teşhis iddiası olmamalıdır. Sistem “erken uyarı / anomali tespiti” olarak anlatılmalıdır.

---

## 15. Teknik Öncelik Sırası

Claude şu sırayla ilerlemelidir:

### Öncelik 1 — Çalıştırılabilirlik

Kod tek komutla açılmalı. Import hataları, eksik klasör ve eksik model hataları sistemi çökertmemeli.

### Öncelik 2 — Ses pipeline

Dosya yükleme, ses okuma, normalize etme, segmentleme, spektrogram üretme çalışmalı.

### Öncelik 3 — Tahmin sistemi

Model varsa gerçek tahmin, model yoksa demo/mock tahmin çalışmalı.

### Öncelik 4 — Arayüz

Gradio paneli sunum seviyesine getirilmeli.

### Öncelik 5 — Loglama

SQLite log sistemi çalışmalı.

### Öncelik 6 — Dokümantasyon

README ve requirements hazırlanmalı.

### Öncelik 7 — Test

Temel fonksiyonlar için basit testler eklenmeli.

---

## 16. Arayüz Metinleri

Arayüzde kullanılabilecek örnek ifadeler:

```text
CİK-SÖR Akustik Erken Uyarı Sistemi

Normal Vokalizasyon
Anormal / Hastalık Şüphesi Taşıyan Vokalizasyon
Gürültü / Analiz Dışı Ses

Güven Skoru
Son Analiz Zamanı
Sistem Durumu
Model Durumu
Mikrofon Durumu
Uyarı Durumu

Anormal vokalizasyon tespit edildi. Veteriner kontrolü önerilir.
Bu sonuç kesin teşhis değildir; erken uyarı amaçlıdır.
```

---

## 17. Kabul Kriterleri

Revizyon sonunda aşağıdaki maddeler sağlanmış olmalıdır:

```text
[ ] python app.py ile arayüz açılıyor.
[ ] Model dosyası olmasa bile sistem demo/mock mode ile açılıyor.
[ ] WAV dosyası yüklenip analiz edilebiliyor.
[ ] MP3/MP4 desteği mümkünse çalışıyor; mümkün değilse düzgün hata veriyor.
[ ] Mikrofondan 5 saniye kayıt alınabiliyor veya mikrofon yoksa düzgün uyarı veriyor.
[ ] Ham ses dalgası görüntüleniyor.
[ ] Spektrogram görüntüleniyor.
[ ] Healthy / Unhealthy / Noise sonucu gösteriliyor.
[ ] Güven skoru gösteriliyor.
[ ] Unhealthy durumunda alarm banner çıkıyor.
[ ] Loglar SQLite’a kaydediliyor.
[ ] Loglar arayüzde görüntüleniyor.
[ ] requirements.txt güncel.
[ ] README mevcut.
[ ] Kod modüler ve anlaşılır.
```

---

## 18. Önemli Sınırlılıklar

Bu aşamada şunlar yapılmamalı veya iddia edilmemeli:

- Kesin hastalık teşhisi iddia edilmemeli.
- Veteriner doğrulaması olmadan “Avian Influenza tespit edildi” gibi net ifadeler kullanılmamalı.
- Sahadan toplanmamış veriyle yüksek doğruluk iddiası yazılmamalı.
- Çevresel sensörler varmış gibi mimari kurulmaya çalışılmamalı.
- Tüm iş tek `app.py` içine gömülmemeli.

---

## 19. Gelecek Sürüm Notları

İleride materyaller ve saha verisi geldiğinde şu geliştirmeler yapılacaktır:

```text
- Gerçek sağlıklı tavuk seslerinin toplanması
- Veteriner gözlemli anormal/hasta seslerin etiketlenmesi
- Noise sınıfının saha mekanik sesleriyle güçlendirilmesi
- Modelin gerçek veriyle yeniden eğitilmesi
- Jetson Nano üzerinde edge inference testi
- TFLite optimizasyonu
- Uzun süreli 7/24 kayıt ve ring buffer sistemi
- Bildirim sistemi
- Web panel veya mobil bildirim entegrasyonu
```

---

## 20. Claude İçin Net Görev

Bu projeyi mevcut haliyle analiz et ve aşağıdaki hedeflere göre revize et:

```text
1. Projeyi tek komutla çalışır hale getir.
2. Kodları modülerleştir.
3. Ses işleme pipeline’ını sağlamlaştır.
4. Gradio arayüzünü sunumda gösterilebilir hale getir.
5. Model yoksa sistemi çökertmeden demo/mock mode ile çalıştır.
6. Model varsa gerçek tahmin yap.
7. Healthy / Unhealthy / Noise sınıflarını standartlaştır.
8. Unhealthy çıktısını “Anormal / hastalık şüphesi” olarak göster.
9. Spektrogram ve ses dalgası görselleştirmelerini ekle.
10. SQLite loglama ekle veya mevcutsa düzelt.
11. requirements.txt ve README.md oluştur/güncelle.
12. Gereksiz, kırık veya tekrar eden kodları temizle.
13. Kodda anlaşılır hata mesajları ve logging kullan.
14. Demo günü internet gerektirmeyecek şekilde sistemi stabil hale getir.
```

Son ürün, gerçek saha verisi gelmeden önce CİK-SÖR’ün yazılımsal mimarisini gösteren, çalışan ve jüri/sunum ortamında güvenle kullanılabilecek bir **V1 Demo prototipi** olmalıdır.
