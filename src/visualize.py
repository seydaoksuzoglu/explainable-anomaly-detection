"""
visualize.py — Faz 1, Adım 1.4: takip sonucunu videoya çizmek.

Her karenin üzerine çizilenler:
  - her nesnenin kutusu + ID numarası (her ID'nin kendi sabit rengi var)
  - ayak noktası (kutunun alt-ortası; ARCHITECTURE.md'deki "konum = ayak noktası" kararı)
  - iz: o ID'nin son 3 saniyedeki ayak noktalarını birleştiren çizgi = yörünge
  - sol üstte kare numarası
  - (--zones verilirse, Faz 3) bölgeler yarı saydam turuncu; ayak noktası bir bölgenin içinde
    olan nesnenin kutusu turuncu ve kalın, etiketinde bölgenin adı
  - (--events verilirse, Adım 3.3b) rules.py'nin ONAYLADIĞI olay süresince nesne kırmızı, kalın
    kutu + kırmızı zeminli "GIRIS"/"BOLGEDE" etiketi; altta alarmın gerekçesi
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
from zones import ZONE_COLOR, check_frame_size, draw_zones, inside, load_zones

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

def draw_size(out_height):
    """Çizim ölçülerinin (çizgi kalınlığı, yazı boyu, nokta) büyütme katsayısı.

    Ölçüler 480 piksel yükseklik için ayarlandı (Ped2'yi 2 kat büyütünce 480 oluyor).
    Büyük karede aynı ölçüler incecik kalır: 1080 piksellik MOT17'de 1 piksellik çizgi,
    video ekrana sığdırılınca neredeyse kaybolur. Bu yüzden ölçüleri yükseklikle orantılı
    büyütüyoruz; küçük karelerde ise küçültmüyoruz (max(1, ...)), yazı okunmaz olmasın.
        Ped2 (480 çıktı) → 1      Avenue (360) → 1      MOT17 (1080) → 2.25
    """
    return max(1.0, out_height / TARGET_HEIGHT)

ALARM_COLOR = (0, 0, 255)   # BGR kırmızı: ONAYLI alarm. Araçlar da kırmızı ama ince kutu ve
                            # zeminsiz etiketle; alarm kalın kutu + kırmızı zeminli etiketle ayrılır.
ALARM_HOLD_SECONDS = 2.0   # alarm şeridi onaydan sonra EN AZ bu kadar ekranda kalır: iz alarm anında
                           # biterse (kadrajdan çıktı / koptu) alarm tek karede görünüp kaybolmasın
EVENT_LABEL = {"giris": "GIRIS", "bolgede_goruldu": "BOLGEDE", "hiz": "HIZ", "yon_ihlali": "YON",
               "oyalanma": "OYALANMA"}


def load_events(path):
    """rules.py'nin olay tablosunu okur → {track_id: [olay, ...]}.

    Neden olayları burada yeniden hesaplamıyoruz? Karar TEK yerde verilir (rules.py), video
    yalnız gösterir: kural değişince video kendiliğinden doğru kalır. Ayrıca durağan iz işareti
    gibi bilgiler izin tamamına bakmayı gerektirir; kare kare ilerleyen video bunu bilemez.
    ID'ler eşleşir çünkü video da yörünge dosyası da aynı takipçiyle (aynı sonuçla) üretildi.
    """
    import pandas as pd                               # yalnız --events verilince lazım
    events = {}
    for e in pd.read_csv(path).to_dict("records"):
        events.setdefault(int(e["track_id"]), []).append(e)
    return events


def alarm_text(tid, e):
    """Alt şerit için kısa gerekçe (cv2 Türkçe harf çizemez → ASCII, ondalık virgüllü)."""
    num = lambda v, k=1: f"{v:.{k}f}".replace(".", ",")
    if e["event"] == "hiz":
        return f"ALARM ID {tid} | hiz {num(e['t_start'])} sn | en yuksek {num(e['peak'], 2)} boy/sn"
    if e["event"] == "oyalanma":
        limit = e["t_alarm"] - e["t_start"]          # alarm = giriş + sınır → sınır bu farktır
        return (f"ALARM ID {tid} | {e['zone']} | oyalanma {num(e['dwell_s'])} sn "
                f"(sinir {num(limit)} sn)")
    if e["event"] == "yon_ihlali":
        return (f"ALARM ID {tid} | {e['zone']} | izinli yon disi {num(e['t_start'])} sn | "
                f"sapma {e['peak']:.0f} derece")
    kind = "giris" if e["event"] == "giris" else "bolgede (iz icerde basladi)"
    return (f"ALARM ID {tid} | {e['zone']} | {kind} {num(e['t_start'])} sn | "
            f"en derin {num(e['max_depth_h'], 2)} boy")


def draw_tracks(frame, frame_no, tracks, trails, scale, zones=None, alarms=None):
    """Bir kareye kutuları, ID'leri, ayak noktalarını ve izleri çizer.

    trails: ID → son ayak noktaları kuyruğu. Bu fonksiyon onu GÜNCELLER
    (yeni noktayı ekler), böylece bir sonraki karede iz bir adım uzar.
    zones : (isteğe bağlı) bölge listesi. Önce bölgeler çizilir ki kutular üstte kalsın.
    alarms: (isteğe bağlı) bu karede ONAYLI olayı süren izler: {track_id: [olay, ...]}.
            Bir izin aynı anda birden çok olayı olabilir (ör. hız + yön); kutuya ilki yazılır,
            alt şeritte hepsi görünür.
    """
    out = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame.copy()
    zones = zones or []
    alarms = alarms or {}
    if zones:
        out = draw_zones(out, zones, scale)

    # Ölçüler çıktı karesinin boyutuna göre (draw_size). k=1 iken eski ölçülerin aynısı.
    k = draw_size(out.shape[0])
    line = max(1, round(k))    # cv2 kalınlıkları tam sayı ister

    for track_id, name, score, x1, y1, x2, y2 in tracks:
        fx, fy = foot_point(x1, y1, x2, y2)
        trails[track_id].append((int(fx * scale), int(fy * scale)))

        is_person = name == "person"
        color = color_for(track_id) if is_person else (0, 0, 255)  # yaya dışı: hep kırmızı
        thick = line if is_person else 2 * line                    # ve kalın, göze çarpsın

        # Ayak noktası bir bölgenin içinde mi? (HAM karar: sınırda titreyen kişi her karede
        # içeri/dışarı gidip gelir — bu titremeyi Adım 3.3'teki debounce süzecek. Burada onu
        # bilerek ham gösteriyoruz ki videoda görülebilsin.)
        in_zones = [z.name for z in zones if inside(z, fx, fy)]
        if in_zones:
            color, thick = ZONE_COLOR, 2 * line                    # bölgedeki nesne turuncu
        alarm = alarms[track_id][0] if track_id in alarms else None
        if alarm is not None:                                      # onaylı olay: turuncuyu ezer
            color, thick = ALARM_COLOR, 3 * line

        # İz: ardışık ayak noktalarını çizgiyle birleştir
        pts = trails[track_id]
        for a, b in zip(pts, list(pts)[1:]):
            cv2.line(out, a, b, color, line)

        # Kutu
        cv2.rectangle(out, (int(x1 * scale), int(y1 * scale)),
                      (int(x2 * scale), int(y2 * scale)), color, thick)
        # Ayak noktası
        cv2.circle(out, pts[-1], round(3 * k), color, -1)   # -1 = içi dolu daire
        # Etiket: yaya için sadece ID, diğerleri için sınıf adı da
        label = f"{track_id}" if is_person else f"{track_id} {name}"
        if in_zones and alarm is None:
            label += " > " + ",".join(in_zones)   # cv2 Türkçe/özel karakter çizemez → düz ASCII
        org = (int(x1 * scale), int(y1 * scale) - round(4 * k))
        if alarm is not None:
            # Kırmızı zemin + beyaz yazı: alarm, kırmızı çizilen araçlardan bir bakışta ayrılsın.
            label = f"{track_id} {EVENT_LABEL.get(alarm['event'], 'ALARM')}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, line)
            cv2.rectangle(out, (org[0], org[1] - th - round(3 * k)), (org[0] + tw + round(4 * k), org[1] + round(3 * k)),
                          ALARM_COLOR, -1)
            cv2.putText(out, label, (org[0] + round(2 * k), org[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k,
                        (255, 255, 255), line)
        else:
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, color, line)

    # Sol üstte kare numarası (beyaz yazı, okunsun diye siyah kenarlı)
    text = f"kare {frame_no}"
    org = (round(10 * k), round(25 * k))  # yazı büyüyünce kenardan uzaklığı da büyüsün
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 0, 0), 4 * line)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (255, 255, 255), 2 * line)

    # Alt şerit: alarm sürdüğü sürece GEREKÇESİ ekranda (açıklanabilirlik). Kişi o karede tespit
    # edilmese bile alarm sürüyorsa şerit kalır — alarm izin değil olayın özelliğidir.
    lines = [alarm_text(tid, e) for tid, evs in sorted(alarms.items()) for e in evs][:3]
    if lines:
        h = round(20 * k) * len(lines) + round(8 * k)
        y0 = out.shape[0] - h
        band = out.copy()
        cv2.rectangle(band, (0, y0), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
        out = cv2.addWeighted(band, 0.6, out, 0.4, 0)        # yarı saydam siyah şerit
        for i, t in enumerate(lines):
            cv2.putText(out, t, (round(8 * k), y0 + round(20 * k) * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * k, (80, 80, 255), line)
    return out


def write_tracking_video(source, out_path, tracker=None, fps=None, zones=None, frame_size=None,
                         events=None):
    """Bir klibi takip edip çizimli videoyu out_path'e yazar.

    fps: verilmezse kaynağın kendi fps'i kullanılır (gerçek hız).
         Elle verilirse video hızlandırılır/yavaşlatılır (sadece gösterim için).
    zones, frame_size: load_zones'un döndürdükleri (isteğe bağlı).
    events: load_events'in döndürdüğü {track_id: [olay]} (isteğe bağlı).
    """
    events = events or {}
    seen_ids = set()
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
            if zones:
                check_frame_size(frame_size, frame)   # yanlış sahne dosyasına karşı uyarı
        # Bu karede onaylı olayı süren izler: alarm anından olay sonuna kadar.
        hold = round(ALARM_HOLD_SECONDS * real_fps)
        active = [(tid, e) for tid, evs in events.items() for e in evs
                  if e["frame_alarm"] <= frame_no <= max(e["frame_end"], e["frame_alarm"] + hold)]
        alarms = {}
        for tid, e in active:
            alarms.setdefault(tid, []).append(e)
        seen_ids.update(t[0] for t in tracks)
        out = draw_tracks(frame, frame_no, tracks, trails, scale, zones, alarms)
        if writer is None:
            # Video yazıcıyı ilk karede açıyoruz, çünkü boyutu (genişlik, yükseklik) ancak
            # ilk çizimden sonra biliyoruz. "mp4v" = .mp4 için yaygın bir sıkıştırma biçimi.
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(out)

    writer.release()   # dosyayı düzgün kapat; yoksa video bozuk kalabilir
    missing = set(events) - seen_ids
    if missing:
        # Olay dosyasındaki ID videoda hiç yok → olaylar başka bir takip çalışmasından (farklı
        # takipçi/ayar) üretilmiş olabilir; kırmızı kutular yanlış kişilere düşer.
        print(f"UYARI: olay dosyasındaki ID'ler videoda görülmedi: {sorted(missing)}. "
              "Olaylar aynı takip ayarlarıyla mı üretildi?")
    speed = f" ({fps / real_fps:.1f}x hızlandırılmış)" if fps != real_fps else ""
    print(f"Video kaydedildi: {out_path} — {frame_no} kare, {fps:g} fps{speed}")


# Kullanım:
#   python src/visualize.py                                       → Test001, gerçek hız
#   python src/visualize.py data/UCSDped2/Test/Test004            → başka bir klip
#   python src/visualize.py benim_videom.mp4                      → gerçek bir video
#   python src/visualize.py data/UCSDped2/Test/Test004 --fps 25   → sunum için hızlandırılmış
#   python src/visualize.py data/UCSDped1/Test/Test011 --zones configs/zones/UCSDped1.json
#   python src/visualize.py data/UCSDped1/Test/Test027 --zones configs/zones/UCSDped1.json \
#       --events outputs/events/UCSDped1_Test027.csv
if __name__ == "__main__":
    # argparse: komut satırını okur. sys.argv'yi elle ayrıştırmaktan daha güvenli ve
    # "python src/visualize.py --help" yazınca kullanım bilgisini kendisi gösterir.
    parser = argparse.ArgumentParser(description="Takip sonucunu videoya çizer.")
    parser.add_argument("source", nargs="?", default="data/UCSDped2/Test/Test001",
                        help="kare klasörü ya da video dosyası")
    parser.add_argument("--fps", type=float, default=None,
                        help="çıktı videonun fps'i (verilmezse kaynağın gerçek fps'i)")
    parser.add_argument("--zones", help="sahne dosyası: bölgeleri ve bölgedeki nesneleri çiz")
    parser.add_argument("--events", help="rules.py olay tablosu: onaylı alarmları kırmızı çiz")
    args = parser.parse_args()
    zones, frame_size = load_zones(args.zones) if args.zones else (None, None)

    Path("outputs").mkdir(exist_ok=True)
    out_path = Path("outputs") / f"tracking_{Path(args.source).stem}.mp4"
    events = load_events(args.events) if args.events else None
    write_tracking_video(args.source, out_path, fps=args.fps, zones=zones, frame_size=frame_size,
                         events=events)
