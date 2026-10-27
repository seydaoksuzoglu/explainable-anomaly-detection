# ROADMAP

Durum: `[ ]` yapılacak · `[~]` devam · `[x]` bitti

## Faz 0 — Kurulum  [x]
- [x] Python ortamı + paketler (ultralytics, opencv-python, shapely, numpy, scikit-learn, sqlalchemy, PySide6)
- [x] Git repo
- [x] UCSD Ped2 indir, yapıyı doğrula → `data/UCSDped2/Train` (16 klip), `Test` (12 klip +
      `TestXXX_gt` piksel maskeleri), kare düzeyi etiketler `Test/UCSDped2.m`
> Çıktı: çalışan ortam + veri yerinde.

## Faz 0b — Ek veri setleri  [x]
> Hoca geri bildirimi (2026-10-20): "birkaç veri seti üzerinde çalış". Stretch → çekirdek.
- [x] UCSD Ped1 (zaten indirildi) → `data/UCSDped1`'e taşı, yapıyı doğrula
- [x] CUHK Avenue indir, yapıyı doğrula → videolar `data/Avenue/testing_videos`, etiketler
      `data/Avenue/testing_label_mask` (resmî ground_truth_demo.zip; 21/21 video, 15324 kare,
      3712 anomali karesi = %24.2, kare sayıları videolarla birebir)
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

## Faz 2 — Öznitelik çıkarma  [x]
- [x] Adım 2.1: yumuşatma (0.5 sn), hız (piksel/sn + boy/sn), yön, kalış süresi → `src/features.py`
- [x] Adım 2.2: ivme (işaretli), yön değişimi (°/sn, işaretli, unwrap), izin tek sınıfı (track_class)
      → `outputs/features/<veriseti>_<klip>.csv`
- [~] Düzeltme: ≤ 1 sn boşluklar hesaptan önce ara değerle doldurulur (boşluk kenarında tek
      taraflı pencere → sahte hız sıçraması). Kod `_handoff/features-gapfill/`'de test edildi.
- [x] hız "saniyede" ölçülsün (veri setlerinin fps'i farklı: 10 / 25 / 30) → time_s'ten
- bölgeye mesafe → Faz 3'e taşındı (bölge poligonu orada tanımlanıyor)
> Çıktı: öznitelik tablosu.

## Faz 3 — Kanal A: Kural motoru  [x]
> Ayrıntılı ölçümler ve gerekçeler DECISIONS.md'de (2026-10-27 / 28 kayıtları).
- [x] 3.1 Bölge tanımı → `src/zones.py`, sahne dosyaları `configs/zones/<veriseti>.json`
      (Ped1 çimen; Avenue çimen + merdiven yön bölgesi + turnike önü oyalanma; MOT17-09 dükkân girişi).
      Tıklama aracı: köşeler + (yön bölgesinde) izinli yön okları.
- [x] 3.2 Bölge sütunları (in_<ad>, dist_<ad> boy cinsinden) + iz yer değiştirmesi / durağan işareti
      → `features.py --zones`; görsel `visualize.py --zones` (bölgedeki nesne turuncu).
- [x] 3.3 Durum makinesi + debounce (histerezis: giriş ≥ 0,15 boy içeride 0,5 sn, çıkış dışarıda 1 sn)
      → `src/rules.py`; videoda onaylı alarm kırmızı + gerekçe şeridi (`visualize.py --events`).
      Ped1 36 klip: kare bazlı 441 alarm → 7 olay (5'i etiketli çimen girişi).
- [x] 3.4a Ortak süreklilik mekanizması (Persistence); hız kuralı (eşik eğitim videolarından,
      Avenue %99 = 0,87 boy/sn → 13 olay, %69'u etiketli anomali anında); yön kuralı (yasak yönde
      ≥ 0,2 boy ilerleme; Avenue merdiven senaryosu 75 inişin 43'ü, çıkanlara 0 alarm).
- [x] 3.4b Oyalanma: loiter bölgesinde kalış > max_dwell_s → alarm = giriş + sınır (Avenue turnike önü).
- [x] 3.5 Bütün kural olayları tek tabloda: `<out-dir>/<sahne>_tum_olaylar.csv` (clip sütunu +
      kural × bölge özeti). Kanal A'nın çıktısı; Faz 5 füzyonu ve Faz 6 olay günlüğü bunu okur.
- Bilinen sınırlılık: kurallar takip kalitesine bağlı (iz parçalanması yön ve oyalanmayı en çok
  etkiliyor) → Ertelenen iş "Tespit ve takip iyileştirme". Yön/giriş kuralları için etiketli senaryo
  videosu Faz 7'de.
> Çıktı: gerekçeli kural alarmları. ✓

## Faz 4 — Kanal B: Öğrenilmiş model  [ ]
- [ ] Çıktı klasör düzeni (kullanıcı önerisi): outputs/<veriseti>/<bölüm: train|test>/
      {trajectories,features,events,videos}/<klip> — Avenue train/test aynı adlı dosyaları birbirinin
      üzerine yazıyor; Faz 4 train ve test özniteliklerini ayrı okuyacağı için önce bu.
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
      Ön ölçüm yapıldı (2026-10-27, motmetrics, yaklaşık): 9 yapılandırma IDF1 0,52–0,54;
      BoT-SORT (ReID'siz) en dengeli. Resmî TrackEval + HOTA ile tekrarlanacak.
      Betikler: `_handoff/takip_olcum/` (run_trackers.py, mot_eval.py)
- [ ] kural vs Isolation Forest vs füzyon karşılaştırması
> Çıktı: sonuç tablosu/grafik (raporun yıldızı).

## Ertelenen iş — Tespit ve takip iyileştirme (hocamın önerisi, 2026-10-27)  [ ]
> Şimdilik YAPILMIYOR; Faz 3 bitince öncelik yeniden değerlendirilecek. Gerekçe ve ölçümler:
> DECISIONS.md "Tespit/takip iyileştirme ertelendi".
Denenebilecek yöntemler (darboğaz sırasına göre):
- [ ] Dedektör: aynı takipçiyle YOLO11s / 11m / 11x ve transformer tabanlı RT-DETR (ultralytics
      içinde, kurulum yok) → recall + IDF1. "YOLO'yu bırakalım mı?" sorusunun cevabı.
- [ ] Gerçek ReID: BoT-SORT / Deep OC-SORT + OSNet (ultralytics "auto" ReID dedektör özniteliği,
      zayıf). Kapanma sonrası kimliği korumayı hedefler.
- [ ] Birleşik tespit+takip (hoca önerisi): CenterTrack (merkez noktası + önceki kareden kayma;
      ReID yok), FairMOT (tespit + kimlik imzası tek ağda), transformer tabanlı MOT (hangisi
      olduğu netleşecek: TrackFormer / MOTR / TransTrack / TransMOT).
      Riskler: eski kod tabanları + DCNv2 derlemesi (torch 2.11 / RTX 50 serisi), hazır modellerin
      MOT17 train ile eğitilmiş olması (aynı dizilerde ölçmek veri sızıntısı), ağır hesap.
Veri seti / sahne:
- [ ] Tesis güvenliğinde kamera genelde yüksekten bakar (hoca). Takip değerlendirmesi yüksek açılı
      sahnelere ağırlık versin: MOT17-04, UCSD Ped1/Ped2; aday: MOT20 (kalabalık, yüksek açı).
      Kanıt: OC-SORT IDF1 MOT17-04 (yüksek açı) 0,585 · MOT17-09 0,563 · MOT17-02 (alçak) 0,391.

## Faz 8 — Stretch (vakit kalırsa)  [ ]
- [ ] LSTM autoencoder + karşılaştırma
- [ ] zorlu koşul (düşük ışık) testi
