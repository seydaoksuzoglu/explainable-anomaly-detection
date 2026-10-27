"""
zones.py — Faz 3, Adım 3.1: bölge (ilgi alanı) tanımı.

Bölge = görüntü üzerinde köşe noktalarıyla tanımlı bir çokgen (poligon). Kamera sabit olduğu
için bir piksel her karede yerdeki aynı noktayı gösterir; ilk karede çizilen bölge videonun
sonuna kadar geçerlidir (DECISIONS: sabit kamera varsayımı).

Bu dosyanın dört işi var:
    1) Sahne dosyasını (JSON) okuyup shapely poligonlarına çevirmek  → load_zones
    2) "Ayak noktası bölgenin içinde mi?" sorusunu cevaplamak        → inside
    3) Öznitelik tablosuna bölge sütunları eklemek (Adım 3.2)         → zone_features
    4) Bölgeyi çizmek / görsel kontrol                                → draw_zones, tıklama aracı

Sahne dosyası (configs/zones/<sahne>.json), veri seti başına BİR tane: bir veri setinin bütün
klipleri aynı sabit kameradan. data/ değil configs/: bölge deneyin parçası, git'te durmalı.
    {
      "scene": "UCSDped1",
      "frame_size": [158, 238],                      ← [yükseklik, genişlik] (çizildiği kare)
      "zones": [
        {"name": "cimen_sag", "type": "restricted", "points": [[x, y], [x, y], ...]}
      ]
    }
type: kuralı belirler (restricted = giriş yasak, loiter = oyalanma; Adım 3.4'te kullanılacak).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon

# NOT: detection (YOLO) burada içe aktarılmıyor; yalnız first_frame içinde, gerektiğinde.
# features.py bölge sütunları için bu dosyayı kullanıyor ve YOLO'yu yüklemesine gerek yok
# (ultralytics'i yüklemek birkaç saniye sürer, öznitelik hesabının ise ona ihtiyacı yok).

ZONE_TYPES = ("restricted", "loiter")
ZONE_COLOR = (0, 140, 255)   # BGR turuncu: takip renkleriyle (yeşil/mavi tonlar) karışmasın


class Zone:
    """Tek bir bölge: adı, türü ve shapely poligonu."""

    def __init__(self, name, zone_type, points):
        if zone_type not in ZONE_TYPES:
            raise ValueError(f"Bilinmeyen bölge türü: {zone_type} (geçerli: {ZONE_TYPES})")
        if len(points) < 3:
            raise ValueError(f"Bölge '{name}' en az 3 köşe ister, {len(points)} verildi")
        self.name = name
        self.type = zone_type
        self.points = [tuple(p) for p in points]
        self.polygon = Polygon(self.points)
        # Kendini kesen poligon (köşeler yanlış sırayla tıklanmış, "papyon" şekli) içeride/dışarıda
        # sorusuna güvenilir cevap vermez → baştan reddet.
        if not self.polygon.is_valid:
            raise ValueError(f"Bölge '{name}' geçersiz (kendini kesiyor olabilir): köşeleri "
                             "çevre boyunca sırayla tıkla")

    def __repr__(self):
        return f"Zone({self.name!r}, {self.type!r}, {len(self.points)} köşe)"


def load_zones(path):
    """Sahne dosyasını okur → (zones listesi, frame_size [yükseklik, genişlik])."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    zones = [Zone(z["name"], z["type"], z["points"]) for z in data["zones"]]
    return zones, data.get("frame_size")


def check_frame_size(frame_size, frame):
    """Bölge başka çözünürlükte çizildiyse uyar: piksel koordinatları o zaman yanlış yere düşer."""
    if frame_size and list(frame_size) != list(frame.shape[:2]):
        print(f"UYARI: bölge {frame_size[0]}x{frame_size[1]} karede çizilmiş, video "
              f"{frame.shape[0]}x{frame.shape[1]}. Yanlış sahne dosyası olabilir.")


def inside(zone, x, y):
    """Nokta(lar) bölgenin içinde mi? Tek sayı ya da dizi (tablonun bütün satırları) alır.

    shapely.contains_xy: noktaları tek tek Point nesnesine çevirmeden, numpy dizisi üzerinde
    toplu çalışır → binlerce satırlık öznitelik tablosu bir çağrıda biter.
    'contains' = kesin içeride; tam sınır çizgisinin ÜSTÜNDEKİ nokta dışarıda sayılır.
    (Sınırda titreyen noktaları Adım 3.3'teki debounce halledecek; burada önemi yok.)
    """
    return shapely.contains_xy(zone.polygon, np.asarray(x, float), np.asarray(y, float))


def zone_features(df, zones):
    """Öznitelik tablosuna her bölge için iki sütun ekler (Adım 3.2):

        in_<ad>   : ayak noktası bölgenin içinde mi (True/False), ham — titreme süzülmemiş.
        dist_<ad> : ayak noktasının bölge SINIRINA uzaklığı, BOY cinsinden, işaretli:
                    içeride eksi, dışarıda artı (−1.5 = sınırdan 1,5 boy içeride).

    Neden boy cinsinden? Hızdaki gibi perspektif: uzaktaki kişi için 5 piksel büyük, yakındaki
    için küçük bir mesafedir. Kutu yüksekliğine (h_s, features.py'nin yumuşatılmış değeri)
    bölünce "sınıra ne kadar yakın" sorusu kişinin kameraya uzaklığından bağımsız olur.
    Neden işaretli? Debounce (3.3) "sınırda titreyen" (−0.1 ile +0.1 arası) nokta ile
    "bölgenin derinine girmiş" nokta arasındaki farkı bu sayıdan görecek.
    """
    df = df.copy()
    # shapely.points: x, y dizilerinden toplu nokta dizisi (tek tek Point kurmadan).
    pts = shapely.points(df["x_foot"].to_numpy(float), df["y_foot"].to_numpy(float))
    for z in zones:
        inn = inside(z, df["x_foot"], df["y_foot"])
        # polygon.boundary = poligonun kenar çizgisi. İçerideki noktanın poligona uzaklığı 0
        # olduğu için (zaten içinde), sınıra uzaklığı kenar çizgisine olan mesafeden ölçüyoruz.
        d = shapely.distance(z.polygon.boundary, pts)
        df[f"in_{z.name}"] = inn
        df[f"dist_{z.name}"] = np.where(inn, -d, d) / df["h_s"].to_numpy(float)
    return df


def draw_zones(frame, zones, scale=1):
    """Bölgeleri karenin bir kopyasına yarı saydam dolgu + kenar + adıyla çizer.

    scale: kare çizimden önce büyütüldüyse (visualize.py küçük kareleri büyütür) aynı oran.
    """
    out = frame.copy()
    overlay = frame.copy()
    for z in zones:
        pts = (np.array(z.points) * scale).round().astype(np.int32)
        cv2.fillPoly(overlay, [pts], ZONE_COLOR)
    # addWeighted: iki görüntüyü harmanlar → dolgu %25 saydam, arkadaki sahne görünür kalır
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    # Çizgi ve yazı ölçüleri kare boyuyla orantılı (visualize.draw_size ile aynı formül):
    # 480 piksellik karede 1, 1080 piksellik MOT17'de 2,25 → büyük karede yazı okunur kalır.
    k = max(1.0, out.shape[0] / 480)
    line = max(1, round(k))
    for z in zones:
        pts = (np.array(z.points) * scale).round().astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=ZONE_COLOR, thickness=2 * line)
        # Adı bölgenin ALT kenarına yakın yaz: sol üst köşe (ilk yöntem) üçgen gibi şekillerde
        # bölgenin dışına, yürüyenlerin üstüne düşüyordu; ortası ise içeride yürüyenin üstüne.
        # x: representative_point (poligonun içinde olması garanti bir nokta), y: en alt kenar.
        p = z.polygon.representative_point()
        bottom = z.polygon.bounds[3]                      # bounds = (minx, miny, maxx, maxy)
        cv2.putText(out, z.name, (int(p.x * scale) - round(30 * k), int(bottom * scale) - round(6 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * k, ZONE_COLOR, line)
    return out


def first_frame(source):
    """Kaynağın ilk karesi (bölge çizmek için)."""
    from detection import iter_frames                # yalnız burada lazım (bkz. dosya başı)
    for _, frame in iter_frames(source):
        return frame
    raise IOError(f"Kaynakta kare yok: {source}")


def draw_tool(source, scene_path, name, zone_type):
    """İlk kareyi pencerede gösterir; fareyle köşeleri tıklatıp bölgeyi sahne dosyasına ekler.

    Sol tık: köşe ekle · Sağ tık / Backspace: son köşeyi sil · Enter: kaydet · Esc: vazgeç
    Küçük kareler (Ped1 158x238) ekranda büyütülerek gösterilir; tıklanan nokta kaydedilirken
    aynı oranla ORİJİNAL piksel koordinatına geri çevrilir.
    """
    frame = first_frame(source)
    scale = max(1, round(600 / frame.shape[0]))      # ~600 piksel yüksekliğe büyüt
    big = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    scene_path = Path(scene_path)
    existing = load_zones(scene_path)[0] if scene_path.exists() else []
    points = []

    def on_mouse(event, x, y, flags, param):
        # OpenCV fare olayında bu fonksiyonu çağırır; x, y büyütülmüş görüntüdeki piksel.
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((round(x / scale, 1), round(y / scale, 1)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    win = f"Bolge: {name}  |  sol tik: kose  sag tik: geri al  Enter: kaydet  Esc: vazgec"
    # WINDOW_NORMAL: pencere boyutu elle ayarlanabilir olsun. WSLg bazen pencereyi ekran
    # dışında ya da sıfır boyutta açar; boyutu ve konumu açıkça vermek bunu önler.
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, big.shape[1], big.shape[0])   # genişlik, yükseklik (piksel)
    cv2.moveWindow(win, 50, 50)                         # ekranın sol üstüne yakın
    cv2.setMouseCallback(win, on_mouse)

    while True:
        view = draw_zones(big, [z for z in existing if z.name != name], scale)
        pts = (np.array(points) * scale).round().astype(np.int32) if points else None
        if pts is not None:
            cv2.polylines(view, [pts], isClosed=len(points) >= 3, color=(0, 0, 255), thickness=2)
            for p in pts:
                cv2.circle(view, tuple(int(v) for v in p), 4, (0, 0, 255), -1)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF                 # 30 ms bekle, basılan tuşu oku
        if key == 27:                                # Esc
            print("Vazgeçildi, bir şey kaydedilmedi.")
            break
        if key == 8 and points:                      # Backspace
            points.pop()
        if key in (13, 10):                          # Enter
            zone = Zone(name, zone_type, points)     # geçersizse burada hata verir
            others = [z for z in existing if z.name != name]   # aynı adlı eski bölgenin yerine
            save_scene(scene_path, others + [zone], frame.shape[:2])
            print(f"Kaydedildi: {scene_path} → {zone}")
            break
    cv2.destroyAllWindows()


def save_scene(path, zones, frame_size):
    """Bölgeleri sahne dosyasına yazar (klasör yoksa oluşturur)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"scene": path.stem, "frame_size": [int(v) for v in frame_size],
            "zones": [{"name": z.name, "type": z.type, "points": [list(p) for p in z.points]}
                      for z in zones]}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def show(source, scene_path, out_path, features=None):
    """Görsel kontrol: bölgeleri ilk kareye çizer. Öznitelik CSV'si verilirse bütün ayak
    noktalarını da koyar: bölge içindekiler kırmızı, dışındakiler yeşil."""
    zones, frame_size = load_zones(scene_path)
    frame = first_frame(source)
    check_frame_size(frame_size, frame)
    scale = max(1, round(480 / frame.shape[0]))
    big = cv2.resize(frame, None, fx=scale, fy=scale)
    out = draw_zones(big, zones, scale)
    if features:
        import pandas as pd                          # yalnız bu seçenekte lazım
        f = pd.read_csv(features)
        any_inside = np.zeros(len(f), bool)
        for z in zones:
            hit = inside(z, f["x_foot"], f["y_foot"])
            print(f"{z.name}: {hit.sum()} / {len(f)} satır içeride, "
                  f"{f.loc[hit, 'track_id'].nunique()} farklı iz")
            any_inside |= hit
        for (x, y), hit in zip(f[["x_foot", "y_foot"]].to_numpy() * scale, any_inside):
            cv2.circle(out, (int(x), int(y)), 1, (0, 0, 255) if hit else (0, 200, 0), -1)
    cv2.imwrite(str(out_path), out)
    print(f"Görsel: {out_path}")


# Kullanım:
#   Bölge çiz (pencere açılır):
#     python src/zones.py draw data/UCSDped1/Test/Test001 configs/zones/UCSDped1.json --name cimen_sag
#   Kontrol görseli (pencere gerekmez):
#     python src/zones.py show data/UCSDped1/Test/Test001 configs/zones/UCSDped1.json
#     python src/zones.py show data/UCSDped2/Test/Test001 configs/zones/UCSDped2.json \
#         --features outputs/features/UCSDped2_Test001.csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bölge çizme ve kontrol aracı.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="ilk kareye fareyle bölge çiz")
    d.add_argument("source"); d.add_argument("scene")
    d.add_argument("--name", required=True, help="bölge adı (aynı ad varsa üzerine yazılır)")
    d.add_argument("--type", default="restricted", choices=ZONE_TYPES)
    s = sub.add_parser("show", help="bölgeleri ilk kareye çizip resim kaydet")
    s.add_argument("source"); s.add_argument("scene")
    s.add_argument("--features", help="öznitelik CSV'si: ayak noktalarını içeride/dışarıda boya")
    s.add_argument("--out", default="outputs/zones_check.jpg")
    args = parser.parse_args()

    if args.cmd == "draw":
        draw_tool(args.source, args.scene, args.name, args.type)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        show(args.source, args.scene, args.out, args.features)
