"""
detection.py - Faz 1, Adım 1: kare okuma + YOLO ile nesne tespiti.

Bu dosyanın iki işi var:
    1) iter_frames(): bir kaynaktan (Ped2 kare klasörü VEYA video dosyası) kareleri sırayla vermek.
    2) Detector:      YOLO modelini bir kere yükleyip, verilen karede nesneleri bulmak.

Takip (kalıcı ID atama) burada YOK - o tracking.py'nin işi (Adım 2).
"""

import configparser         # MOT17'nin seqinfo.ini dosyasını okumak için (Python'un kendi modülü)
from pathlib import Path    # dosya yollarıyla OS'ten bağımsız çalışmak için

import cv2
from ultralytics import YOLO

# Ped2 kareleri .tif; ilerde başka veri setleri .jpg/.png olabilir.
IMAGE_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}

# COCO sınıf numaraları (YOLO'nun eğitildiği veri seti 80 sınıf).
# Sadece 'person' almıyoruz çünkü Ped2'deki anomaliler tam olarak bisiklet, araba,
# kaykay gibi yaya-dışı nesneler. 
TRACKED_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    7: "truck",
    36: "skateboard",
}

def iter_frames(source):
    """Kaynaktaki kareleri (frame_no, kare) çiftleri olarak sırayla üretir.
    
    source:
        - klasör ise --> içindeki resim dosyaları sırasıyla (001.tif, 002.tif, ...)
          MOT17 dizi klasörü ise (ör. MOT17-02-FRCNN) --> kareler onun img1/ alt klasöründen.
        - dosya ise --> video dosyası (mp4, avi...), cv2.VideoCapture ile okunur.

    frame_no 1'den başlar. Neden 0 değil? Ped2'nin etiket dosyası (UCSDped2.m) kareleri
    1'den sayıyor (ör. gt_frame = [61:180]). Aynı sayımı kullanırsak Faz 7'de etiketlerle
    eşleştirirken bir kare kayma hatası yapmayız.

    'yield' kullandığımız için bu bir generator: kareleri tek tek, ihtiyaç oldukça okur.
    Tüm videoyu belleğe yüklemez.
    """
    source = Path(source)

    if source.is_dir():
        # MOT17 yapısı: MOT17-02-FRCNN/ altında img1/ (kareler), gt/, seqinfo.ini.
        # Kaynak olarak DİZİ klasörünü veriyoruz (img1'i değil): böylece dizi adı
        # (MOT17-02-FRCNN) CSV/video adlarına girer ve seqinfo.ini'ye (fps) yanında ulaşılır.
        # Kareleri ise img1/'den okuyoruz. Ped klasörlerinde img1 yok → bu satır bir şey yapmaz.
        if (source / "img1").is_dir():
            source = source / "img1"
        # sorted(): dosya sisteminin verdiği sıra garanti değil biz ad sırasına zorluyoruz.
        # Ped2 adları sıfırlarla dolu (001, 002...) olduğundan ad sırası = zaman sırası.
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for frame_no, path in enumerate(files, start=1):
            # cv2.imread varsayılanı IMREAD_COLOR: gri .tif'i de 3 kanallı (BGR) yapar.
            # YOLO 3 kanallı görüntü beklediğinden bu dönüşüm bize bedavaya gelir.
            frame = cv2.imread(str(path))
            if frame is None:
                raise IOError(f"Kare okunamadı: {path}")
            yield frame_no, frame
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise IOError(f"Video açılamadı: {source}")
        frame_no = 0
        try:
            while True:
                ok, frame = cap.read()   # ok=False -> video bitti
                if not ok:
                    break
                frame_no += 1
                yield frame_no, frame
        finally:
            cap.release()  # döngü yarıda kesilse bile video dosyasını kapat

def source_fps(source, default=10.0):
    """Kaynağın kare hızını (fps = saniyedeki kare sayısı) döndürür.

    - Video dosyası (mp4, avi...) kendi fps'ini içinde saklar → oradan okuruz.
    - MOT17 dizi klasörü fps'i seqinfo.ini dosyasında yazar (frameRate=30) → oradan okuruz.
    - Düz kare klasörü (Ped2 gibi) fps bilgisi taşımaz → varsayılanı döndürürüz.
      Ped2 ve Ped1 saniyede 10 kare kaydedildiği için varsayılan 10.

    Neden lazım? Videoyu kaynağın gerçek fps'iyle yazmazsak hızı bozulur: 30 fps'lik bir
    telefon videosunu 10 fps ile yazarsak 3 kat ağır çekim oynar. Faz 2'de de hızı
    "saniyede" ölçmek için fps'e ihtiyacımız olacak.
    """
    source = Path(source)
    if source.is_dir():
        # Eskiden klasör için hep 10 dönüyordu → MOT17 (30 fps) 3 kat yanlış olurdu:
        # takipçi 3 sn yerine 1 sn bekler, time_s ve video hızı bozulurdu.
        seqinfo = source / "seqinfo.ini"
        if seqinfo.is_file():
            # .ini dosyası "[Bölüm]" başlıkları ve "anahtar=değer" satırlarından oluşur;
            # configparser bunu sözlük gibi okur: info["Sequence"]["frameRate"] → "30"
            info = configparser.ConfigParser()
            info.read(seqinfo)
            return float(info["Sequence"]["frameRate"])
        return default
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)   # okunamazsa 0 döner
    cap.release()
    return fps if fps > 0 else default

class Detector:
    """YOLO modelini sarmalayan küçük sınıf.
    
    Neden sınıf kullanıyoruz? Model bir kere yüklenir (yavaş işlem, GPU'ya taşınır), sonra binlerce
    karede tekrar tekrar kullanılır. Ayarlar (imgsz, conf, sınıflar) da tek yerde durur.
    """

    def __init__(self, weights="weights/yolo11s.pt", imgsz=1280, conf=0.1):
        # Dosya yoksa ultralytics onu bu yolla otomatik indirir.
        # GPU varsa kendiliğinden onu kullanır.
        self.model = YOLO(weights)
        # imgsz=1280: Ped2 kareleri 240x360 ve yayalar çok küçük. YOLO görüntüyü bu boyuta
        # büyütüp öyle bakar; küçük nesneler daha çok piksel kaplar -> daha iyi yakalanır.
        # (Denememizde 640'a göre daha çok kişi ve daha yüksek güven skoru verdi.)
        self.imgsz = imgsz
        # conf: bu güven skorunun altındaki tespitler atılır.
        # 0.1 gibi DÜŞÜK bir eşik seçtik, çünkü takip katmanı düşük güvenli tespitlere de
        # ihtiyaç duyar (ByteTrack/OC-SORT'un ikinci eşleştirme turu 0.1–0.25 arasını kullanır).
        # Eskiden model.track() bunu kendisi 0.1'e çekiyordu; artık tespiti biz yaptığımız
        # için bu ayarı bilerek kendimiz veriyoruz. "Kesin" tespit isteyen yer kendisi süzer.
        self.conf = conf
        # Sözlüğün anahtarları -> [0, 1, 2, 3, 7, 36]. YOLO sadece bu sınıfları döndürür.
        self.classes = list(TRACKED_CLASSES)

    def detect(self, frame):
        """Bir karedeki nesneleri sayı tablosu (numpy dizisi) olarak döndürür.

        Biçim: (N, 6) boyutlu dizi; her satır bir tespit:
            [x1, y1, x2, y2, güven, sınıf_no]
        (x1, y1) sol üst, (x2, y2) sağ alt köşe; piksel cinsinden, orijinal kare ölçeğinde.
        N = 0 olabilir (karede hiçbir şey yoksa boş dizi döner).

        Neden artık liste değil de dizi? Takip katmanı (tracking.py) tespitleri bu sayısal
        biçimde ister. Sınıf ADI ("person") yerine sınıf NUMARASI (0) tutuyoruz; ada
        sadece insanlara gösterirken TRACKED_CLASSES ile çeviriyoruz.
        """
        # predict() bir liste döndürür (aynı anda birden çok görüntü verilebildiği için).
        # Tek kare verdik → ilk elemanı alıyoruz. verbose=False: her karede konsola yazmasın.
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf,
            classes=self.classes, verbose=False,
        )[0]

        # result.boxes.data zaten tam istediğimiz biçimde: (N, 6) = x1,y1,x2,y2,güven,sınıf.
        # Değerler GPU'da duruyor → .cpu() ile ana belleğe, .numpy() ile numpy dizisine.
        return result.boxes.data.cpu().numpy()


def draw_detections(frame, detections, scale=2):
    """Tespitleri, karenin büyütülmüş bir kopyasına çizer.

    Önce büyütüp sonra çiziyoruz: 240x360 karede yazılar okunmaz kadar küçük kalıyor.
    Kopya üzerinde çalıştığımız için orijinal kare bozulmaz.
    """
    out = cv2.resize(frame, None, fx=scale, fy=scale)
    for x1, y1, x2, y2, score, cls_id in detections:
        name = TRACKED_CLASSES[int(cls_id)]   # numara → ad (0 → "person")
        # Koordinatlar orijinal ölçekte → büyütme oranıyla çarp; çizim tam sayı piksel ister.
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        color = (0, 255, 0) if name == "person" else (0, 0, 255)  # yaya yeşil, diğerleri kırmızı
        cv2.rectangle(out, p1, p2, color, 1)
        cv2.putText(out, f"{name} {score:.2f}", (p1[0], p1[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


# Bu blok sadece dosya doğrudan çalıştırılınca çalışır:  python src/detection.py
# Başka bir dosya "from detection import Detector" dediğinde ÇALIŞMAZ.
if __name__ == "__main__":
    clip = "data/UCSDped2/Test/Test001"   # etikete göre 61. kareden itibaren anomali var
    detector = Detector()
    Path("outputs").mkdir(exist_ok=True)

    for frame_no, frame in iter_frames(clip):
        # Şimdilik sadece iki kareye bakıyoruz: 1 (normal) ve 100 (anormal bölge)
        if frame_no not in (1, 100):
            continue
        dets = detector.detect(frame)
        # Ekrana/resme sadece güvenli tespitleri (≥ 0.25) koyalım; 0.1–0.25 arası takip içindir.
        # dets[:, 4] = tüm satırların 5. sütunu (güven). Koşul, satırları süzen bir maske üretir.
        dets = dets[dets[:, 4] >= 0.25]
        names = [TRACKED_CLASSES[int(c)] for c in dets[:, 5]]
        print(f"kare {frame_no}: {len(dets)} tespit ->",
            {n: names.count(n) for n in set(names)})
        cv2.imwrite(f"outputs/detect_Test001_{frame_no:03d}.jpg",
                    draw_detections(frame, dets))



