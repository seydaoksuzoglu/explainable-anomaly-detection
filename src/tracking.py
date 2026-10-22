"""
tracking.py — Faz 1, Adım 2: ByteTrack ile her nesneye kalıcı kimlik (ID) verme.

detection.py her kareye ayrı ayrı bakar: "bu karede 24 kişi var" der ama
"kare 5'teki şu kişi, kare 4'teki hangi kişiydi?" sorusunu cevaplayamaz.
Takipçi bu soruyu cevaplar: aynı kişiye her karede aynı ID'yi verir.
ID olmadan yörünge olmaz; yörünge olmadan davranış (hız, oyalanma...) olmaz.
"""

from collections import Counter   # ID başına kaç karede görüldüğünü saymak için

from detection import Detector, TRACKED_CLASSES, iter_frames


class Tracker:
    """Detector'ın modelini ByteTrack ile birlikte çalıştıran sınıf.

    Not: Ultralytics'te tespit ve takip TEK çağrıda yapılır: model.track(...) önce YOLO ile
    kutuları bulur, sonra takipçiye verir. Bu yüzden burada detector.detect()'i çağırmıyoruz;
    Detector'dan sadece yüklü modeli ve ayarlarını (imgsz, sınıflar) ödünç alıyoruz.
    Böylece model ayarları tek yerde (detection.py) durur.
    """

    def __init__(self, detector=None, tracker_cfg="bytetrack.yaml"):
        # Dışarıdan bir Detector verilmezse varsayılan ayarlarla yenisini oluştur.
        self.detector = detector if detector is not None else Detector()
        # Takipçi bir ayar dosyasıyla seçilir. "botsort.yaml" yazmak takipçiyi değiştirmeye
        # yeter → CLAUDE.md'deki "takipçi modüler/swappable" kuralı bu tek parametre.
        self.tracker_cfg = tracker_cfg

    def track_clip(self, source):
        """Bir klibi baştan sona takip eder; her kare için (frame_no, kare, izler) verir.

        izler = [(track_id, sınıf_adı, güven, x1, y1, x2, y2), ...]

        Bu da bir generator (yield): iter_frames gibi kareleri tek tek işler.
        """
        for frame_no, frame in iter_frames(source):
            result = self.detector.model.track(
                frame,
                # persist: takipçinin hafızasını bir sonraki çağrıya taşı.
                #   Kare 1'de False → yeni, boş bir takipçi kurulur ve ID'ler 1'den başlar.
                #   Sonraki karelerde True → takipçi önceki kareleri hatırlar, ID'ler korunur.
                # Böylece her klip temiz başlar; Test001'in son ID'si Test002'ye taşmaz.
                persist=frame_no > 1,
                tracker=self.tracker_cfg,
                imgsz=self.detector.imgsz,
                classes=self.detector.classes,
                # conf'u BİLEREK vermiyoruz: ultralytics takipte onu 0.1'e çeker.
                # ByteTrack'in farkı, düşük güvenli (0.1–0.25) tespitleri de kullanmasıdır:
                # örneğin kalabalıkta yarı kapanmış birinin güveni düşer ama hâlâ oradadır;
                # ByteTrack onu mevcut bir izle eşleyip kimliğini kaybetmez.
                verbose=False,
            )[0]

            tracks = []
            # boxes.id: takipçinin verdiği ID'ler. Karede hiç iz yoksa None gelir.
            if result.boxes.id is not None:
                for track_id, xyxy, score, cls_id in zip(result.boxes.id.int().tolist(),
                                                         result.boxes.xyxy.tolist(),
                                                         result.boxes.conf.tolist(),
                                                         result.boxes.cls.int().tolist()):
                    x1, y1, x2, y2 = xyxy
                    tracks.append((track_id, TRACKED_CLASSES[cls_id], score, x1, y1, x2, y2))

            yield frame_no, frame, tracks


# Doğrudan çalıştırınca:  python src/tracking.py
if __name__ == "__main__":
    clip = "data/UCSDped2/Test/Test001"
    tracker = Tracker()

    seen = Counter()      # track_id → kaç karede görüldü
    track_class = {}      # track_id → sınıf adı (en son görülen)

    for frame_no, frame, tracks in tracker.track_clip(clip):
        for track_id, name, *_ in tracks:   # *_ : kalan alanlar (güven, kutu) şimdilik lazım değil
            seen[track_id] += 1
            track_class[track_id] = name

        # Birkaç karede hangi ID'lerin görüldüğünü yazdır
        if frame_no in (1, 2, 100):
            ids = sorted(t[0] for t in tracks)
            print(f"kare {frame_no}: {len(tracks)} iz, ID'ler: {ids}")

    # --- Özet: takip ne kadar sağlam? ---
    lengths = list(seen.values())
    short = [tid for tid, n in seen.items() if n < 10]
    print(f"\nToplam farklı ID: {len(seen)}")
    print(f"Bir ID ortalama {sum(lengths) / len(lengths):.0f} karede görüldü (klip {frame_no} kare)")
    print(f"10 kareden kısa süren ID sayısı: {len(short)}")
    bikes = [tid for tid, c in track_class.items() if c != "person"]
    for tid in bikes:
        print(f"Yaya olmayan iz: ID {tid} ({track_class[tid]}), {seen[tid]} karede görüldü")
