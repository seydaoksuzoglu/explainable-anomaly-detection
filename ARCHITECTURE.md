# ARCHITECTURE

## Genel akış
```
Video
  → YOLO (tespit, detection.py)
  → takipçi: OC-SORT varsayılan (takip, kalıcı ID, tracking.py)
  → yörünge CSV'si (trajectory.py)
  → öznitelik çıkarma (yörüngeden)
        ├── Kanal A: kurallar (zones.py bölgeleri + rules.py durum makinesi; giriş, hız, yön, oyalanma)
        └── Kanal B: öğrenilmiş model (sadece normalle eğitilmiş tek-sınıf)
  → geç füzyon (ağırlıklı)
  → açıklanabilir alarm
  → olay günlüğü (SQLite) + arayüz (PySide6)
```

## Modüller (dosyalar)
| Dosya | Görev |
|---|---|
| `detection.py` | YOLO ile nesne tespiti |
| `tracking.py` | tespitlerden kimlik atama; takipçi swappable (OC-SORT varsayılan, ByteTrack), bekleme süresi saniye cinsinden |
| `trajectory.py` | takip sonucunu klip başına yörünge CSV'sine yazar (`outputs/trajectories/<veriseti>_<klip>.csv`) |
| `visualize.py` | kutu + ID + ayak noktası + iz çizili video; --zones bölgeler, --events onaylı alarmlar (kırmızı + gerekçe şeridi) |
| `zones.py` | bölge (poligon) tanımı: sahne JSON'u, ayak noktası içeride mi, sınıra uzaklık; tıklama aracı (köşeler + izinli yön okları) |
| `features.py` | yörüngeden öznitelik: hız, ivme, yön, yön değişimi, bölgeye mesafe, kalış süresi |
| `rules.py` | Kanal A: ortak süreklilik durum makinesi (Persistence) + kurallar: yasak bölgeye giriş, hız, yön ihlali, oyalanma; gerekçe cümlesi; eşik kalibrasyonu (--calibrate); klip başına ve tek tabloda olaylar |
| `model.py` | Kanal B: Isolation Forest (normalle eğitilir); stretch: LSTM-AE |
| `fusion.py` | iki kanal skorunu ağırlıklı birleştir + gerekçe üret |
| `db.py` | SQLAlchemy ORM, olay günlüğü (SQLite) |
| `ui.py` | PySide6 arayüz; video işleme QThread'de |
| `evaluate.py` | Ped2 test + gt ile metrikler, kanal karşılaştırması |

## Veri biçimleri
- **Yörünge (CSV):** `frame, time_s, track_id, class, conf, x_foot, y_foot, x1, y1, x2, y2`
  (koordinatlar orijinal kare pikseli; time_s = (frame − 1) / fps)
- **Öznitelik:** `track_id, frame, speed, accel, heading, heading_change, dist_to_zone, dwell`
- **Kural olayı (Kanal A, `rules.py`):** `clip, zone, zone_type, track_id, class, event, t_start, t_alarm, t_end,
  dwell_s, frame_start, frame_alarm, frame_end, peak, max_depth_h, closed, reason`
  (event ∈ giris, bolgede_goruldu, hiz, yon_ihlali, oyalanma)
- **Olay (DB, Faz 6):** `timestamp, track_id, class, event_type, score, reason, zone_id`

## Anahtar tasarım ilkeleri (detay → DECISIONS.md)
- Sabit kamera → piksel = bölge; kamera hareketi telafisi yok.
- Nesne konumu = **ayak noktası** (bbox alt-ortası), yer düzlemi için doğru.
- Hibrit: kural (açıklanabilir, net olaylar) + öğrenilmiş model (kuralla yakalanmayan sapma).
- Açıklanabilirlik **yapıca**: hangi kanal/öznitelik tetikledi → gerekçe cümlesi.
- Eğitim **sadece normalle** (unsupervised); anomali **yalnızca testte** görülür.
