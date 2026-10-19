# ROADMAP

Durum: `[ ]` yapılacak · `[~]` devam · `[x]` bitti

## Faz 0 — Kurulum  [x]
- [x] Python ortamı + paketler (ultralytics, opencv-python, shapely, numpy, scikit-learn, sqlalchemy, PySide6)
- [x] Git repo
- [x] UCSD Ped2 indir, yapıyı doğrula → `data/UCSDped2/Train` (16 klip), `Test` (12 klip +
      `TestXXX_gt` piksel maskeleri), kare düzeyi etiketler `Test/UCSDped2.m`
> Çıktı: çalışan ortam + veri yerinde.

## Faz 1 — Algı → yörünge çıkarma  [ ]
- [ ] YOLO + ByteTrack'i bir test klibinde çalıştır
- [ ] Her nesne: ID + ayak noktası + kare kare konum kaydı
- [ ] Kutular + ID + iz çizili görsel video
> Çıktı: yörünge dosyası (frame, id, class, x, y) + görsel.

## Faz 2 — Öznitelik çıkarma  [ ]
- [ ] hız, ivme, yön, yön değişimi, bölgeye mesafe, kalış süresi
> Çıktı: öznitelik tablosu.

## Faz 3 — Kanal A: Kural motoru  [ ]
- [ ] shapely poligon (ilk karede bölge tanımı)
- [ ] durum makinesi (dışarıda ↔ içeride) + debounce
- [ ] oyalanma süresi, ters yön
> Çıktı: gerekçeli kural alarmları.

## Faz 4 — Kanal B: Öğrenilmiş model  [ ]
- [ ] Ped2 train (sadece normal) öznitelikleriyle Isolation Forest eğit
- [ ] test'te anomali skoru + hangi öznitelik saptı
> Çıktı: anomali skoru.

## Faz 5 — Füzyon + açıklama  [ ]
- [ ] iki kanalın skorunu ağırlıklı birleştir
- [ ] gerekçe cümlesi üret
> Çıktı: birleşik açıklanabilir alarm.

## Faz 6 — Kayıt + arayüz  [ ]
- [ ] SQLite + SQLAlchemy ORM olay günlüğü
- [ ] PySide6 arayüz (video, bölge çizme, canlı sayım, olay listesi + filtre)
> Çıktı: çalışan masaüstü uygulaması.

## Faz 7 — Değerlendirme  [ ]
- [ ] Ped2 test + gt ile AUROC, precision/recall, yanlış alarm oranı
- [ ] kural vs Isolation Forest vs füzyon karşılaştırması
> Çıktı: sonuç tablosu/grafik (raporun yıldızı).

## Faz 8 — Stretch (vakit kalırsa)  [ ]
- [ ] LSTM autoencoder + karşılaştırma
- [ ] ikinci dataset (genelleme testi)
- [ ] tracker karşılaştırması (ByteTrack vs BoT-SORT → yanlış alarm)
- [ ] zorlu koşul (düşük ışık) testi
