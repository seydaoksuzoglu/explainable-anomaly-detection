# ARCHITECTURE

## Genel akış
```
Video
  → YOLO (tespit)
  → ByteTrack (takip, kalıcı ID)
  → öznitelik çıkarma (yörüngeden)
        ├── Kanal A: kurallar (shapely + durum makinesi)
        └── Kanal B: öğrenilmiş model (sadece normalle eğitilmiş tek-sınıf)
  → geç füzyon (ağırlıklı)
  → açıklanabilir alarm
  → olay günlüğü (SQLite) + arayüz (PySide6)
```

## Modüller (dosyalar)
| Dosya | Görev |
|---|---|
| `detection.py` | YOLO ile nesne tespiti |
| `tracking.py` | ByteTrack ile kimlik atama (takipçi swappable) |
| `features.py` | yörüngeden öznitelik: hız, ivme, yön, yön değişimi, bölgeye mesafe, kalış süresi |
| `rules.py` | Kanal A: shapely poligon + durum makinesi (giriş, oyalanma, ters yön) + debounce |
| `model.py` | Kanal B: Isolation Forest (normalle eğitilir); stretch: LSTM-AE |
| `fusion.py` | iki kanal skorunu ağırlıklı birleştir + gerekçe üret |
| `db.py` | SQLAlchemy ORM, olay günlüğü (SQLite) |
| `ui.py` | PySide6 arayüz; video işleme QThread'de |
| `evaluate.py` | Ped2 test + gt ile metrikler, kanal karşılaştırması |

## Veri biçimleri
- **Yörünge:** `frame, track_id, class, x_foot, y_foot`
- **Öznitelik:** `track_id, frame, speed, accel, heading, heading_change, dist_to_zone, dwell`
- **Olay (DB):** `timestamp, track_id, class, event_type, score, reason, zone_id`

## Anahtar tasarım ilkeleri (detay → DECISIONS.md)
- Sabit kamera → piksel = bölge; kamera hareketi telafisi yok.
- Nesne konumu = **ayak noktası** (bbox alt-ortası), yer düzlemi için doğru.
- Hibrit: kural (açıklanabilir, net olaylar) + öğrenilmiş model (kuralla yakalanmayan sapma).
- Açıklanabilirlik **yapıca**: hangi kanal/öznitelik tetikledi → gerekçe cümlesi.
- Eğitim **sadece normalle** (unsupervised); anomali **yalnızca testte** görülür.
