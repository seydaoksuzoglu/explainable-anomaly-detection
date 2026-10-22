"""
visualize.py — Faz 1, Adım 1.4: takip sonucunu videoya çizmek.

Her karenin üzerine çizilenler:
  - her nesnenin kutusu + ID numarası (her ID'nin kendi sabit rengi var)
  - ayak noktası (kutunun alt-ortası; ARCHITECTURE.md'deki "konum = ayak noktası" kararı)
  - iz: o ID'nin son 3 saniyedeki ayak noktalarını birleştiren çizgi = yörünge
  - sol üstte kare numarası
Sonra kareler birleştirilip outputs/ altına .mp4 olarak kaydedilir.

Neden ayrı bir dosya? Çizim, takipten ayrı bir iş. Faz 6'da arayüz de bu çizim
fonksiyonunu kullanacak; tracking.py'yi değiştirmeden.

Birden çok veri seti için iki şey kaynağa göre ayarlanır (sabit DEĞİL):
  - fps:   video, kaynağın gerçek hızında oynasın (Ped2 10, Avenue 25, MOT17 30 fps)
  - ölçek: küçük kareler büyütülür, büyük kareler olduğu gibi kalır
"""

import argparse                              # komut satırı argümanlarını okumak için
from collections import defaultdict, deque   # deque: sabit uzunlukta kuyruk (iz için)
from pathlib import Path

import cv2
import numpy as np

from detection import source_fps
from tracking import Tracker

TRAIL_SECONDS = 3      # iz kaç saniye geriye uzansın (kare sayısı fps'e göre hesaplanır)
TARGET_HEIGHT = 480    # küçük kareleri en az bu yüksekliğe yaklaşacak kadar büyüt


def color_for(track_id):
    """Her ID'ye sabit bir renk verir: aynı kişi videonun başından sonuna aynı renkte.

    HSV renk uzayında 'ton' (hue) değerini ID'den türetiyoruz; 37 ile çarpmak, ardışık
    ID'lerin (5, 6, 7...) birbirine benzemeyen renkler almasını sağlar.
    """
    # Kırmızı tonlar (0-20 ve 160-179) yaya dışı nesnelere ayrıldı; yayalar bu aralığa
    # düşmesin diye tonu sadece 25-154 arasından seçiyoruz (130 farklı ton).
    hue = 25 + (track_id * 37) % 130
    # cvtColor resim bekler; tek renklik 1x1 piksellik bir "resim" verip HSV → BGR çeviriyoruz.
    hsv = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in hsv[0, 0])


def foot_point(x1, y1, x2, y2):
    """Kutunun alt-ortası: kişinin yere bastığı nokta (konumumuz bu)."""
    return (x1 + x2) / 2, y2


def scale_for(frame):
    """Çizim için büyütme oranı: küçük kareyi büyüt, büyüğe dokunma.

    Ped2 (240 yükseklik)  → 480/240 = 2  → 2 kat büyütülür
    Avenue (360)          → 480/360 ≈ 1  → olduğu gibi
    MOT17 (1080)          → 480/1080 ≈ 0 → max(1, ...) sayesinde 1, yani olduğu gibi
    """
    return max(1, round(TARGET_HEIGHT / frame.shape[0]))   # shape[0] = yükseklik


def draw_tracks(frame, frame_no, tracks, trails, scale):
    """Bir kareye kutuları, ID'leri, ayak noktalarını ve izleri çizer.

    trails: ID → son ayak noktaları kuyruğu. Bu fonksiyon onu GÜNCELLER
    (yeni noktayı ekler), böylece bir sonraki karede iz bir adım uzar.
    """
    out = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame.copy()

    for track_id, name, score, x1, y1, x2, y2 in tracks:
        fx, fy = foot_point(x1, y1, x2, y2)
        trails[track_id].append((int(fx * scale), int(fy * scale)))

        is_person = name == "person"
        color = color_for(track_id) if is_person else (0, 0, 255)  # yaya dışı: hep kırmızı
        thick = 1 if is_person else 2                              # ve kalın, göze çarpsın

        # İz: ardışık ayak noktalarını çizgiyle birleştir
        pts = trails[track_id]
        for a, b in zip(pts, list(pts)[1:]):
            cv2.line(out, a, b, color, 1)

        # Kutu
        cv2.rectangle(out, (int(x1 * scale), int(y1 * scale)),
                      (int(x2 * scale), int(y2 * scale)), color, thick)
        # Ayak noktası
        cv2.circle(out, pts[-1], 3, color, -1)   # -1 = içi dolu daire
        # Etiket: yaya için sadece ID, diğerleri için sınıf adı da
        label = f"{track_id}" if is_person else f"{track_id} {name}"
        cv2.putText(out, label, (int(x1 * scale), int(y1 * scale) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Sol üstte kare numarası (beyaz yazı, okunsun diye siyah kenarlı)
    text = f"kare {frame_no}"
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def write_tracking_video(source, out_path, tracker=None, fps=None):
    """Bir klibi takip edip çizimli videoyu out_path'e yazar.

    fps: verilmezse kaynağın kendi fps'i kullanılır (gerçek hız).
         Elle verilirse video hızlandırılır/yavaşlatılır (sadece gösterim için).
    """
    tracker = tracker if tracker is not None else Tracker()
    real_fps = source_fps(source)
    fps = fps if fps is not None else real_fps

    # İz uzunluğu saniye cinsinden sabit; kare sayısı fps'e göre değişir:
    # 10 fps → 30 kare, 25 fps → 75 kare. Hepsi "son 3 saniye" demek.
    trail_len = round(TRAIL_SECONDS * real_fps)
    # defaultdict: daha önce görülmemiş bir ID istenince otomatik boş kuyruk oluşturur.
    # deque(maxlen=...): sınır aşılınca en eski nokta kendiliğinden düşer.
    trails = defaultdict(lambda: deque(maxlen=trail_len))
    writer = None
    scale = None

    for frame_no, frame, tracks in tracker.track_clip(source):
        if scale is None:
            scale = scale_for(frame)   # ölçeği ilk karenin boyutundan bir kere belirle
        out = draw_tracks(frame, frame_no, tracks, trails, scale)
        if writer is None:
            # Video yazıcıyı ilk karede açıyoruz, çünkü boyutu (genişlik, yükseklik) ancak
            # ilk çizimden sonra biliyoruz. "mp4v" = .mp4 için yaygın bir sıkıştırma biçimi.
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(out)

    writer.release()   # dosyayı düzgün kapat; yoksa video bozuk kalabilir
    speed = f" ({fps / real_fps:.1f}x hızlandırılmış)" if fps != real_fps else ""
    print(f"Video kaydedildi: {out_path} — {frame_no} kare, {fps:g} fps{speed}")


# Kullanım:
#   python src/visualize.py                                       → Test001, gerçek hız
#   python src/visualize.py data/UCSDped2/Test/Test004            → başka bir klip
#   python src/visualize.py benim_videom.mp4                      → gerçek bir video
#   python src/visualize.py data/UCSDped2/Test/Test004 --fps 25   → sunum için hızlandırılmış
if __name__ == "__main__":
    # argparse: komut satırını okur. sys.argv'yi elle ayrıştırmaktan daha güvenli ve
    # "python src/visualize.py --help" yazınca kullanım bilgisini kendisi gösterir.
    parser = argparse.ArgumentParser(description="Takip sonucunu videoya çizer.")
    parser.add_argument("source", nargs="?", default="data/UCSDped2/Test/Test001",
                        help="kare klasörü ya da video dosyası")
    parser.add_argument("--fps", type=float, default=None,
                        help="çıktı videonun fps'i (verilmezse kaynağın gerçek fps'i)")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    out_path = Path("outputs") / f"tracking_{Path(args.source).stem}.mp4"
    write_tracking_video(args.source, out_path, fps=args.fps)
