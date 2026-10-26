# ROADMAP

Durum: `[ ]` yapılacak · `[~]` devam · `[x]` bitti

## Faz 0 — Kurulum  [x]
- [x] Python ortamı + paketler (ultralytics, opencv-python, shapely, numpy, scikit-learn, sqlalchemy, PySide6)
- [x] Git repo
- [x] UCSD Ped2 indir, yapıyı doğrula → `data/UCSDped2/Train` (16 klip), `Test` (12 klip +
      `TestXXX_gt` piksel maskeleri), kare düzeyi etiketler `Test/UCSDped2.m`
> Çıktı: çalışan ortam + veri yerinde.

## Faz 0b — Ek veri setleri  [ ]
> Hoca geri bildirimi (2026-10-20): "birkaç veri seti üzerinde çalış". Stretch → çekirdek.
- [x] UCSD Ped1 (zaten indirildi) → `data/UCSDped1`'e taşı, yapıyı doğrula
- [~] CUHK Avenue indir, yapıyı doğrula (videolar `data/Avenue`'da; etiket dosyası eksik)
- [x] MOT17 indir (yalnız sabit kameralı sekanslar), yapıyı doğrula → `data/MOT17/train/`
      MOT17-02/04/09-FRCNN (600/1050/525 kare, 30 fps, 1920x1080; 62/83/26 yaya ID)
- [ ] ShanghaiTech Campus indir, yapıyı doğrula — ERTELENDİ (2026-10-27: çok büyük; vakit kalırsa)
> Çıktı: dört ek veri seti `data/` altında, kare/video okuma her birinde çalışıyor.

## Faz 1 — Algı → yörünge çıkarma  [x]
- [x] YOLO + ByteTrack'i bir test klibinde çalıştır
- [x] Her nesne: ID + ayak noktası + kare kare konum kaydı → `src/trajectory.py`,
      `outputs/trajectories/<veriseti>_<klip>.csv` (satır/ID sayısı tracking.py ile birebir)
- [x] Kutular + ID + iz çizili görsel video
- [x] Tespit ve takip katmanlarını ayır (hoca): OC-SORT varsayılan, bekleme süresi saniye cinsinden
- [x] Video kaynağın gerçek fps'iyle yazılsın + boyuta göre ölçek (hoca: "oynatma yavaş")
- [x] Hızlı hareket testi: Ped1 + Ped2 + Avenue'nun tüm test klipleri işlendi (69 video,
      `outputs/<veri seti>/`). Bulgu: Avenue'da iz ömrü kısa (~1.9 sn) → takipçi
      tamponu fps'e göre ayarlanmalı (yapıldı: bekleme saniye cinsinden).
      MOT17-09 (OC-SORT, 2026-10-27, gözle): önü kapanmayan kişide ID kopmuyor, kısa
      kapanmalarda korunuyor (ByteTrack'ten iyi görünüyor); kalabalık geçişte bozuluyor:
      ID 2 kare 117'de kapandı, o sırada kayıp olan ID 4 onun yerine eşlendi (kimlik takası).
      Büyük karelerde çizim ölçüleri kare boyuyla orantılı yapıldı (visualize.draw_size).
> Çıktı: yörünge dosyası (frame, id, class, x, y) + görsel.

## Faz 2 — Öznitelik çıkarma  [ ]
- [ ] hız, ivme, yön, yön değişimi, bölgeye mesafe, kalış süresi
- [ ] hız "saniyede" ölçülsün (veri setlerinin fps'i farklı: 10 / 25 / 30)
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
- [ ] aynı ölçüm Ped1, Avenue, ShanghaiTech üzerinde (genelleme)
- [ ] Takipçi karşılaştırması (çekirdeğe alındı 2026-10-27): MOT17 sabit kamera dizilerinde
      (02, 04, 09) IDF1, MOTA, HOTA; ByteTrack / OC-SORT / BoT-SORT / Deep OC-SORT ...
      + Avenue'da vekil ölçütler (ID sayısı, kapsama) → seçilen takipçinin gerekçesi
- [ ] kural vs Isolation Forest vs füzyon karşılaştırması
> Çıktı: sonuç tablosu/grafik (raporun yıldızı).

## Faz 8 — Stretch (vakit kalırsa)  [ ]
- [ ] LSTM autoencoder + karşılaştırma
- [ ] zorlu koşul (düşük ışık) testi
