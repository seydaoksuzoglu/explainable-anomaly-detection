"""
tracking.py — Faz 1: tespitlerden kalıcı kimlik (ID) üretme. TAKİP KATMANI.

detection.py her kareye ayrı ayrı bakar: "bu karede 24 kişi var" der ama
"kare 5'teki şu kişi, kare 4'teki hangi kişiydi?" sorusunu cevaplayamaz.
Takipçi bu soruyu cevaplar: aynı kişiye her karede aynı ID'yi verir.
ID olmadan yörünge olmaz; yörünge olmadan davranış (hız, oyalanma...) olmaz.

Katmanlar AYRI (hoca önerisi, DECISIONS.md):
    kare → [detection.py: Detector.detect] → tespitler → [bu dosya: takipçi] → izler
Takipçi YOLO'yu hiç bilmez; sadece tespit tablosunu görür. Böylece:
  - her takipçi AYNI tespitlerle çalışır → karşılaştırma adil olur,
  - dedektör ve takipçi birbirinden bağımsız değiştirilebilir.

Çıktı biçimi SABİT (sonraki fazlar sadece bunu görür):
    izler = [(track_id, sınıf_adı, güven, x1, y1, x2, y2), ...]
"""

import argparse
from collections import Counter   # ID başına kaç karede görüldüğünü saymak için

from ultralytics.engine.results import Boxes             # tespitleri takipçinin beklediği kaba koyar
from ultralytics.trackers.track import TRACKER_MAP       # ad → takipçi sınıfı ("ocsort" → OCSORT)
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml          # "ocsort.yaml" adını dosya yoluna çevirir

from detection import Detector, TRACKED_CLASSES, iter_frames, source_fps

# Avenue'da ölçtük: OC-SORT, ByteTrack'e göre %25 daha az ID kopması verdi (kapsama %97).
# Geçici varsayılan; MOT17'de IDF1 ile kesinleşecek.
DEFAULT_TRACKER = "ocsort"
# Kaybolan bir kişi kaç SANİYE beklensin? (Kare cinsinden değil: fps veri setine göre değişir.)
DEFAULT_BUFFER_SECONDS = 3.0
# Şimdilik görünüş (ReID) KULLANMAYAN takipçiler. ReID'li olanlar (botsort, deepocsort,
# tracktrack) ek girdi ister; onları gerçek bir ReID modeliyle ayrı bir adımda ekleyeceğiz.
SUPPORTED_TRACKERS = ("bytetrack", "ocsort", "fasttrack")


class Tracker:
    """Tespit katmanının çıktısını alıp ID veren takip katmanı."""

    def __init__(self, detector=None, tracker_type=DEFAULT_TRACKER,
                 buffer_seconds=DEFAULT_BUFFER_SECONDS):
        if tracker_type not in SUPPORTED_TRACKERS:
            raise ValueError(f"Desteklenen takipçiler: {SUPPORTED_TRACKERS}, verilen: {tracker_type}")
        # Dışarıdan bir Detector verilmezse varsayılan ayarlarla yenisini oluştur.
        self.detector = detector if detector is not None else Detector()
        self.tracker_type = tracker_type
        self.buffer_seconds = buffer_seconds

    def _new_tracker(self, fps):
        """Ayarları okuyup TAZE bir takipçi nesnesi kurar.

        Ayarlar ultralytics'in hazır YAML dosyasından gelir (ör. ocsort.yaml); biz sadece
        bekleme süresini (track_buffer) videonun fps'ine göre yeniden yazarız:
            3 sn × 10 fps (Ped2)   = 30 kare
            3 sn × 25 fps (Avenue) = 75 kare
        Eskiden bu ayar her videoda 30 kareydi → Avenue'da sadece 1.2 saniye ediyordu.
        """
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{self.tracker_type}.yaml")))
        cfg.track_buffer = max(1, round(self.buffer_seconds * fps))
        return TRACKER_MAP[self.tracker_type](args=cfg)

    def track_clip(self, source):
        """Bir klibi baştan sona takip eder; her kare için (frame_no, kare, izler) verir.

        Bu da bir generator (yield): iter_frames gibi kareleri tek tek işler.
        """
        # Her klip için YENİ bir takipçi: önceki klibin izleri taşınmaz, ID'ler 1'den başlar.
        # (Eskiden bunu model.track(persist=...) ile dolaylı yapıyorduk; şimdi açıkça yapıyoruz.)
        tracker = self._new_tracker(source_fps(source))

        for frame_no, frame in iter_frames(source):
            # --- Katman 1: tespit ---
            dets = self.detector.detect(frame)           # (N, 6): x1,y1,x2,y2,güven,sınıf_no

            # --- Katman 2: takip ---
            # Takipçi tespitleri ultralytics'in Boxes kabında ister (.xywh, .conf, .cls
            # özelliklerini oradan okur). frame.shape[:2] = (yükseklik, genişlik).
            out = tracker.update(Boxes(dets, frame.shape[:2]), frame)

            # out: (M, 8) dizi; her satır bir iz: [x1, y1, x2, y2, track_id, güven, sınıf_no, idx]
            # idx = bu izin eşleştiği tespitin sırası (bize lazım değil; _ ile atlıyoruz).
            # Sabit çıktı biçimimize çeviriyoruz; numpy sayılarını sıradan Python sayısına
            # (int/float) çeviriyoruz ki sonraki adımlarda (CSV, veritabanı) sorun çıkmasın.
            tracks = [
                (int(tid), TRACKED_CLASSES[int(cls_id)], float(score),
                 float(x1), float(y1), float(x2), float(y2))
                for x1, y1, x2, y2, tid, score, cls_id, _ in out
            ]
            yield frame_no, frame, tracks


def summarize(tracker, clip):
    """Bir klibi takip edip özet sayıları döndürür (takipçi karşılaştırması için)."""
    seen = Counter()    # track_id → kaç karede görüldü
    for _, _, tracks in tracker.track_clip(clip):
        for track_id, *_ in tracks:   # *_ : kalan alanlar şimdilik lazım değil
            seen[track_id] += 1
    return seen


# Kullanım:
#   python src/tracking.py                                              → Ped2 Test001, OC-SORT
#   python src/tracking.py data/Avenue/testing_videos/15.avi --tracker bytetrack
#   python src/tracking.py data/Avenue/testing_videos/15.avi --tracker ocsort --buffer 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bir klibi takip edip özet istatistik verir.")
    parser.add_argument("source", nargs="*", default=["data/UCSDped2/Test/Test001"],
                        help="bir veya birden çok kare klasörü / video dosyası")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, choices=SUPPORTED_TRACKERS)
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER_SECONDS,
                        help="kaybolan iz kaç saniye beklensin")
    args = parser.parse_args()

    tracker = Tracker(tracker_type=args.tracker, buffer_seconds=args.buffer)
    total_ids = total_frames = short = 0
    for clip in args.source:
        seen = summarize(tracker, clip)
        total_ids += len(seen)
        total_frames += sum(seen.values())
        short += sum(1 for n in seen.values() if n < 10)

    print(f"Takipçi: {args.tracker}, bekleme: {args.buffer} sn, klip sayısı: {len(args.source)}")
    print(f"Toplam farklı ID:        {total_ids}")
    print(f"İzlenen kişi-kare:       {total_frames}")
    print(f"10 kareden kısa iz oranı: %{100 * short / max(1, total_ids):.0f}")
