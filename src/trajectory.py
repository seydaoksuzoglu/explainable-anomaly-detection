"""
trajectory.py - Faz 1, Adım 1.3: takip sonucunu yörünge dosyasına (CSV) yazmak.

Şimdiye kadar takip sonucu sadece ekranda/videoda görülüyordu; programın geri kalanı onu
saklamıyordu. Faz 2 (öznitelik: hız, yön, kalış süresi...) bir yörünge tablosu ister;
"ID 7, kare 120'de, şu noktadaydı." Bu dosya o tabloyu üretir.

Neden ayrı bir adım ve dosya kullandım?
    - Takip yavaş (YOLO her karede çalışır); öznitelik/kural denemeleri ise hızlı olmalı.
    Takibi BİR kere yapıp CSV'ye yazarsak, Faz 2-4'te aynı videoyu tekrar tekrar takip 
    etmeden, sadece CSV'yi okuyarak çalışırız.
    - CSV düz metin: Excel'de açılır, gözle kontrol edilir, rapora tablo olarak ekleyeceğiz.

Bir satır = bir nesnenin bir karedeki konumu:
    frame, time_s, track_id, class, conf, x_foot, y_foot, x1, y1, x2, y2
"""

import argparse
import csv
from pathlib import Path

from detection import source_fps
from tracking import DEFAULT_BUFFER_SECONDS, DEFAULT_TRACKER, SUPPORTED_TRACKERS, Tracker
from visualize import foot_point    # ayak noktası TEK yerde tanımlı kalsın (çizim ile aynı)

# Sütunlar ARCHITECTURE.md'deki çekirdek biçim (frame, track_id, class, x_foot, y_foot)
# + olarak bilerek bazı sütunlar ekledim.
#   time_s : saniye cinsinden zaman. Veri setlerinin fps'i farklı (10 / 25 / 30); Faz 2'de
#            hızı "saniyede piksel" ölçmek için kare numarası tek başına yetmez.
#   conf   : tespit güveni. Düşük güvenli (0.1–0.25) izleri sonradan süzebilmek için.
#   x1..y2 : kutunun kendisi. Kutu yüksekliği kişinin kameraya uzaklığını gösterir (uzaktaki
#            kişi küçük görünür → aynı hız daha az piksel eder); ileride hızı buna göre
#            normalleştirmek ve yanlış "skateboard" gibi sabit kutuları ayıklamak için lazım.

COLUMNS = ["frame", "time_s", "track_id", "class", "conf",
           "x_foot", "y_foot", "x1", "y1", "x2", "y2"]

def default_csv_name(source):
    """Kaynaktan çakışmayan bir dosya adı üretir.

    Sadece klip adı yetmez: örneğin "Test001" hem Ped1'de hem Ped2'de var. Bu yüzden iki üst
    klasörün adını başa ekliyoruz:
        data/UCSDped2/Test/Test001             → UCSDped2_Test001.csv
        data/Avenue/testing_videos/01.avi      → Avenue_01.csv
    (.stem = uzantısız ad: "01.avi" → "01"; klasör için adın kendisi.)
    """
    source = Path(source)
    return f"{source.parents[1].name}_{source.stem}.csv"

def write_trajectory(source, out_path, tracker):
    """Bir klibi takip eder, her izi satır satır CSV'ye yazar. Satır ve ID sayısını döndürür.

    Satırları takip ilerledikçe yazıyoruz ki (hepsini bir listeye toplamıyoruz) uzun bir videoda
    bellek dolmaz, program yarıda kesilse bile yazılan kısım dosyada kalır.
    """
    fps = source_fps(source)
    n_rows = 0
    ids = set()     # görülen farklı ID'ler (özet için)

    # newline="": csv modülü satır sonlarını kendisi yönetir; vermezsek Windows'ta her
    # satırın arasına boş satır girer. encoding: Türkçe karakterli yollar/sınıflar sorun çıkarmasın.
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)       # ilk satır: başlıklar

        for frame_no, _, tracks in tracker.track_clip(source):
            # Kare 1 → 0.0 sn. (frame_no 1'den başlıyor; detection.iter_frames'e bakın.)
            time_s = (frame_no - 1) / fps
            for track_id, name, score, x1, y1, x2, y2 in tracks:
                fx, fy = foot_point(x1, y1, x2, y2)
                # Piksel değerlerini 1 ondalığa yuvarlıyoruz: daha hassası anlamsız
                # (tespit kutusu zaten birkaç piksel oynar) ve dosya gereksiz büyür.
                writer.writerow([frame_no, round(time_s, 3), track_id, name, round(score, 3),
                                 round(fx, 1), round(fy, 1),
                                 round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
                n_rows += 1
                ids.add(track_id)
    return n_rows, len(ids)


# Kullanım:
#   python src/trajectory.py                                   → Ped2 Test001
#   python src/trajectory.py data/Avenue/testing_videos/01.avi data/Avenue/testing_videos/06.avi
#   python src/trajectory.py data/UCSDped2/Test/Test0* --tracker bytetrack
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Klipleri takip edip yörünge CSV'si yazar.")
    parser.add_argument("source", nargs="*", default=["data/UCSDped2/Test/Test001"],
                        help="bir veya birden çok kare klasörü / video dosyası")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, choices=SUPPORTED_TRACKERS)
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER_SECONDS,
                        help="kaybolan iz kaç saniye beklensin")
    parser.add_argument("--out-dir", default="outputs/trajectories",
                        help="CSV'lerin yazılacağı klasör")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)   # parents=True: outputs/ yoksa onu da kur

    # Tek Tracker (model bir kere yüklenir); track_clip her klipte takipçiyi sıfırlar.
    tracker = Tracker(tracker_type=args.tracker, buffer_seconds=args.buffer)
    for source in args.source:
        # Ped2'de Test001_gt gibi maske klasörleri de var; glob (Test0*) onları da yakalar.
        if source.endswith("_gt"):
            continue
        out_path = out_dir / default_csv_name(source)
        n_rows, n_ids = write_trajectory(source, out_path, tracker)
        print(f"{out_path}: {n_rows} satır, {n_ids} farklı ID")
