"""
features.py — Faz 2: yörüngeden davranış öznitelikleri çıkarmak.

Girdi : trajectory.py'nin yazdığı yörünge CSV'si (bir satır = bir nesnenin bir karedeki konumu).
Çıktı : aynı satırlar + öznitelik sütunları. Kural kanalı (Faz 3) ve Isolation Forest (Faz 4)
        bu tabloyu okur; videoya bir daha dokunmaz.

Adım 2.1: yumuşatma, hız (piksel/sn ve boy/sn), yön, kalış süresi.
Adım 2.2: ivme, yön değişimi, izin tek sınıfı (track_class).
Düzeltme : kısa boşluklar hesaptan önce ara değerle doldurulur (_fill_gaps).
Adım 3.2 : izin toplam yer değiştirmesi (track_extent_h) ve durağan nesne işareti
           (track_static); --zones verilirse bölge sütunları (zones.zone_features).

Üç temel kural (DECISIONS.md):
  1. Önce yumuşat, sonra türev al. Ayak noktası kutu titredikçe birkaç piksel oynar;
     hız = konumun türevi olduğu için bu titremeyi büyütür.
  2. Zaman farkı HER ZAMAN time_s'ten. Kişinin önü kapanınca takipçi ID'yi korur ama o
     karelerde satır yazılmaz; kare numarasıyla hesaplasak boşluk tek adım sanılır, hız şişer.
  3. Hız iki birimde: piksel/sn ve boy/sn (kutu yüksekliğine bölünmüş). Uzaktaki kişi küçük
     görünür; boy/sn bu perspektif etkisini büyük ölçüde giderir.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Yumuşatma penceresi SANİYE cinsinden: 10, 25 ve 30 fps'te aynı süreyi kapsasın.
# 0.5 sn: titremeyi bastıracak kadar uzun, koşmaya başlama gibi gerçek değişimi
# silmeyecek kadar kısa (insan adımı ~0.5 sn).
SMOOTH_SECONDS = 0.5
# Bundan kısa izlerde türev güvenilmez (10 karede yumuşatma + türev anlamlı sonuç vermez).
# tracking.py'deki "kısa iz" tanımıyla aynı sayı.
MIN_TRACK_FRAMES = 10
# Bu hızın altında kişi "duruyor" sayılır ve yönü tanımsızdır (NaN): duran birinin
# "hareket yönü" titremeden ibarettir, her karede rastgele döner. 0.1 boy/sn ≈ 17 cm/sn.
MIN_MOVING_SPEED_H = 0.1
# Bundan kısa boşluklar (dedektörün kişiyi birkaç kare kaçırması) hesaptan önce ara değerle
# doldurulur. Daha uzun boşlukta kişinin ne yaptığını bilemeyiz (durmuş, dönmüş olabilir);
# düz çizgi uydurmak tahmin olur → oraya dokunmuyoruz, zaman farkı yine time_s'ten doğru.
MAX_GAP_SECONDS = 1.0
# İz bütün ömrü boyunca bundan az yer değiştirdiyse "durağan nesne" sayılır (boy cinsinden).
# Neden? Dedektör bazen direk/tabela gibi sabit nesneleri "person" sanıyor (Ped2 Test001
# ID 21: 162 kare boyunca 3 piksel, ~0,2 boy). Bunlar oyalanma kuralında sahte alarm üretir.
# DİKKAT: "duran izi at" DEĞİL — oyalanan gerçek insan da durur. Fark: bölgede oyalanan insan
# oraya YÜRÜYEREK gelmiştir, izinin toplam yer değiştirmesi büyüktür. Durağan nesne ise
# videonun başından sonuna yerinden hiç kıpırdamaz. Satır silinmez, yalnız işaretlenir.
# Yer değiştirme TEK BAŞINA yetmez: kadraj kenarında 2-3 sn görünüp çıkan yürüyen kişi de
# bu kısa sürede ancak ~0,4 boy yol alır (Ped2 Test001 ID 3, 10, 12). Bu yüzden ikinci şart:
# izin medyan hızı da "duruyor" eşiğinin (MIN_MOVING_SPEED_H) altında olmalı. Yürüyenler
# 0,17+ boy/sn, durağan nesneler 0,01-0,05 boy/sn.
STATIC_EXTENT_H = 0.5


def load_trajectory(path):
    """Yörünge CSV'sini okur; her izi kendi içinde zamana göre sıralar."""
    df = pd.read_csv(path)
    # Önce ID'ye, sonra kareye göre sırala: bir izin satırları alt alta ve zaman sırasında olsun.
    # (CSV kare sırasıyla yazıldı; ID'ler karışık.) Sonraki tüm hesaplar bu sıraya dayanır.
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def _smooth(series, time_s):
    """Bir sütunu, zamanda ortalanmış SMOOTH_SECONDS'lık pencerede ortalar.

    Pencere kare sayısıyla değil süreyle tanımlı: pandas'a indeks olarak zamanı veriyoruz
    (Timedelta), "500ms" diyoruz. center=True: pencere [t − 0.25, t + 0.25] sn — geçmiş ve
    gelecek eşit tartılır, yörünge zamanda kaymaz. min_periods=1: izin başında/sonunda
    pencere yarım kalsa da eldeki noktalarla ortalama al (NaN üretme).
    """
    idx = pd.to_timedelta(time_s.to_numpy(), unit="s")
    window = f"{round(SMOOTH_SECONDS * 1000)}ms"
    smoothed = pd.Series(series.to_numpy(), index=idx).rolling(window, center=True,
                                                               min_periods=1).mean()
    return smoothed.to_numpy()


def _fill_gaps(g):
    """Bir izin kısa boşluklarını doğrusal ara değerle doldurur (yalnız HESAP için).

    Sorun: yumuşatma penceresi zamanda ortalanmış (±0.25 sn). Boşluğun hemen yanındaki
    karede pencerenin bir yarısı boş kalır → ortalama tek taraflı olur, yumuşatılmış konum
    geride kalır ya da öne fırlar → sahte hız düşüşü/sıçraması (Ped2 Test001 ID 139: gerçek
    ~22 px/sn iken boşluk kenarında 6.3 ve 41). Boşluğu doldurunca pencere hep iki taraflı.

    Dönen tablo: kareler f0..f1 (kısa boşluklar dolu, uzun boşluklar yine eksik);
    '_real' sütunu: satır gerçek mi (True) yoksa ara değerle mi üretildi (False).
    """
    # Bu izin fps'i: time_s = (frame − 1) / fps olduğu için iki uçtan geri hesaplanır.
    # (İz en az MIN_TRACK_FRAMES satırlı → süre sıfır olamaz.)
    f0, f1 = g["frame"].iloc[0], g["frame"].iloc[-1]
    fps = (f1 - f0) / (g["time_s"].iloc[-1] - g["time_s"].iloc[0])
    max_gap = round(MAX_GAP_SECONDS * fps)          # 10 fps → 10 kare, 30 fps → 30 kare

    # reindex: ilk kareden son kareye HER kare için bir satır; eksik kareler NaN olur.
    cols = ["time_s", "x_foot", "y_foot", "y1", "y2"]
    full = g.set_index("frame")[cols].reindex(range(f0, f1 + 1))
    missing = full["time_s"].isna()

    # Her boşluğun uzunluğu: ardışık NaN'lara aynı numarayı verip (run) sayıyoruz.
    # missing != missing.shift(): durum her değiştiğinde True → cumsum ile koşu numarası.
    run = (missing != missing.shift()).cumsum()
    gap_len = missing.groupby(run).transform("sum")
    keep = ~missing | (gap_len <= max_gap)          # gerçek satırlar + kısa boşluklar

    # interpolate: NaN'ları iki yanındaki gerçek değerler arasında düz çizgiyle doldurur.
    # Kareler eşit aralıklı olduğundan doğrusal ara değer zamanda da doğru (time_s dahil).
    full = full.interpolate()
    full["_real"] = ~missing
    return full[keep].reset_index()                 # frame yeniden sütun olur


def _track_features(g):
    """TEK bir izin satırlarına öznitelik sütunlarını ekler (g = o izin tablosu)."""
    out = g.copy()
    # Hesap, kısa boşlukları doldurulmuş tabloda yapılır; sonuçlar sadece gerçek satırlara
    # geri yazılır (en altta). Böylece tablo ve satır sayısı değişmez.
    g = _fill_gaps(g)
    t = g["time_s"]

    # --- 1) Yumuşatma ---
    # Kutu yüksekliği de titrer (ve hızı ona böleceğiz) → onu da yumuşatıyoruz.
    g["x_s"] = _smooth(g["x_foot"], t)
    g["y_s"] = _smooth(g["y_foot"], t)
    g["h_s"] = _smooth(g["y2"] - g["y1"], t)

    # --- 2) Hız vektörü (vx, vy): konumun zamana göre türevi ---
    # np.gradient(x, t): her noktada türevi komşu noktalardan hesaplar (ortadaki noktalarda
    # bir önceki ve bir sonrakine bakar). İkinci argüman olarak zamanları verdiğimiz için
    # aralıklar eşit olmak zorunda değil → kapanma boşlukları doğru hesaplanır (kural 2).
    tt = t.to_numpy()
    vx = np.gradient(g["x_s"].to_numpy(), tt)
    vy = np.gradient(g["y_s"].to_numpy(), tt)

    # --- 3) Hız (büyüklük) iki birimde ---
    g["speed_px"] = np.hypot(vx, vy)              # √(vx² + vy²), piksel/sn
    # boy/sn: yayalar için gerçekten "kendi boyu/sn" (Ped2'de yürüyenlerin medyanı ~0.27).
    # Yaya DIŞI sınıflarda kutu insanı değil nesneyi çerçeveler (bisiklet kutusu yayadan kısa)
    # → orada "kendi kutusu/sn"dur; yayalarla karşılaştırırken speed_px daha dürüst.
    g["speed_h"] = g["speed_px"] / g["h_s"]

    # --- 4) Yön (derece, 0–360) ---
    # Görüntüde y AŞAĞI doğru artar; matematikte yukarı. −vy ile çevirip alışık olduğumuz
    # açıyı elde ediyoruz: 0° = sağa, 90° = ekranda yukarı, 180° = sola, 270° = aşağı.
    # arctan2 −180..180 verir; % 360 ile 0..360'a taşıyoruz.
    heading = np.degrees(np.arctan2(-vy, vx)) % 360
    # Duran kişinin yönü anlamsız (titremenin yönü) → NaN (bilinmiyor).
    moving = g["speed_h"] >= MIN_MOVING_SPEED_H
    g["heading"] = np.where(moving, heading, np.nan)

    # --- 5) İvme: hızın zamana göre türevi (İŞARETLİ) ---
    # + = hızlanıyor, − = yavaşlıyor. "Aniden koşmaya başladı" = büyük pozitif ivme.
    # İvme ikinci türev: konumdaki titreme iki kez büyür. Bu yüzden hızı da türevden önce
    # aynı pencereyle bir kez daha yumuşatıyoruz (kural 1, ikinci kez).
    # Birimler: piksel/sn² ve boy/sn².
    g["accel_px"] = np.gradient(_smooth(g["speed_px"], t), tt)
    g["accel_h"] = np.gradient(_smooth(g["speed_h"], t), tt)

    # --- 6) Yön değişimi: yönün zamana göre türevi (derece/sn, İŞARETLİ) ---
    # + = saat yönünün tersine döndü (ekranda sola), − = saat yönünde (sağa). Zikzak = işareti
    # sık değişen büyük değerler; düz yürüyüş ≈ 0.
    # Tuzak: açı 360'ta başa sarar. 359° → 1° gerçekte 2°'lik dönüş ama düz fark −358° çıkar.
    # np.unwrap bu sıçramaları giderir (359, 361, 363...) → türev doğru olur. unwrap radyan
    # ister; bu yüzden NaN'sız ham açıyı radyan olarak alıp sonra dereceye çeviriyoruz.
    angle = np.unwrap(np.arctan2(-vy, vx))
    turn_rate = np.degrees(np.gradient(angle, tt))
    # Duran kişide yön tanımsız → yön değişimi de tanımsız.
    g["heading_change"] = np.where(moving, turn_rate, np.nan)

    # --- 7) Kalış süresi: iz ilk görüldüğünden beri geçen süre (sn) ---
    # Bölgeye özel kalış (oyalanma) Faz 3'te, bölge tanımlanınca hesaplanacak.
    g["dwell_s"] = tt - tt[0]

    # --- 8) İzin toplam yer değiştirmesi (boy) ve durağan nesne işareti ---
    # Yumuşatılmış yörüngenin kapladığı kutunun köşegeni: x'te ve y'de en uç noktalar arası
    # (np.ptp = max − min) → hypot ile köşegen. Boyun medyanına bölünür (perspektif).
    # Yumuşatılmış konum kullanılıyor: ham konumun titremesi duran nesneyi "hareketli" göstermesin.
    extent = np.hypot(np.ptp(g["x_s"]), np.ptp(g["y_s"])) / np.median(g["h_s"])

    # Ara değerli satırları at, hesaplanan sütunları gerçek satırlara yaz. İki tablo da
    # kare sırasında → gerçek satırlar birebir aynı sırada eşleşir.
    real = g["_real"].to_numpy()
    for col in ["x_s", "y_s", "h_s", "speed_px", "speed_h", "heading",
                "accel_px", "accel_h", "heading_change", "dwell_s"]:
        out[col] = g.loc[real, col].to_numpy()
    # İz düzeyinde sabitler: izin her satırına aynı değer yazılır.
    out["track_extent_h"] = extent
    # İki şart birlikte: az yer değiştirdi VE (gerçek satırlarda) tipik hızı "duruyor" düzeyinde.
    out["track_static"] = (extent < STATIC_EXTENT_H) and \
        (np.median(out["speed_h"]) < MIN_MOVING_SPEED_H)
    return out


def compute_features(df):
    """Tüm izler için öznitelikleri hesaplar. Kısa izler tablodan çıkarılır."""
    # Her izin kaç satırı var? transform("size"): sonucu her satıra yayar (ID'nin satır sayısı).
    n = df.groupby("track_id")["frame"].transform("size")
    df = df[n >= MIN_TRACK_FRAMES].copy()

    # İzin TEK sınıfı: sınıf her karede dedektörden ayrı gelir ve bir iz içinde değişebilir
    # (Ped2 Test001 ID 153: 7 kare person, 3 kare bicycle). Kurallar "bu iz bisiklet mi?"
    # sorusuna kare kare değişmeyen bir cevap ister → izin en sık görülen sınıfı (mode).
    # Eşitlikte mode alfabetik ilkini verir (her çalıştırmada aynı sonuç). Kare bazlı
    # 'class' sütunu da kalıyor; track_class onun hemen yanına ekleniyor.
    track_class = df.groupby("track_id")["class"].transform(lambda s: s.mode().iloc[0])
    df.insert(df.columns.get_loc("class") + 1, "track_class", track_class)
    # groupby(...).apply: tabloyu ID'ye göre parçalara böler, her parçaya _track_features
    # uygular, sonuçları alt alta birleştirir. Her iz kendi geçmişiyle hesaplanır;
    # bir kişinin konumu başka birinin hızına karışmaz.
    parts = [_track_features(g) for _, g in df.groupby("track_id", sort=False)]
    if not parts:
        return df.iloc[0:0]   # hiç uzun iz yoksa boş tablo
    out = pd.concat(parts, ignore_index=True)
    # Kare sırasına geri dön (girdi CSV'si gibi): kural motoru zamanda ilerleyerek okuyacak.
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)


# Kullanım:
#   python src/features.py outputs/trajectories/UCSDped2_Test001.csv
#   python src/features.py outputs/trajectories/*.csv
#   python src/features.py outputs/trajectories/UCSDped1_Test0*.csv --zones configs/zones/UCSDped1.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yörünge CSV'lerinden öznitelik tablosu üretir.")
    parser.add_argument("trajectory", nargs="+", help="bir veya birden çok yörünge CSV'si")
    parser.add_argument("--out-dir", default="outputs/features",
                        help="öznitelik CSV'lerinin yazılacağı klasör")
    parser.add_argument("--zones", help="sahne dosyası (configs/zones/<veriseti>.json): "
                        "verilirse her bölge için in_<ad> ve dist_<ad> sütunları eklenir")
    args = parser.parse_args()

    zones = None
    if args.zones:
        from zones import load_zones, zone_features     # yalnız bölge istenince lazım
        zones, _ = load_zones(args.zones)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.trajectory:
        df = load_trajectory(path)
        feats = compute_features(df)
        if zones and len(feats):                        # boş tabloda (hiç uzun iz yok) atla
            feats = zone_features(feats, zones)
        out_path = out_dir / Path(path).name     # aynı ad: UCSDped2_Test001.csv
        # float_format: ondalıkları 3 basamağa kısalt (dosya küçük, gözle okunur kalsın)
        feats.to_csv(out_path, index=False, float_format="%.3f")
        n_dropped = df["track_id"].nunique() - feats["track_id"].nunique()
        n_static = feats.loc[feats["track_static"], "track_id"].nunique() if len(feats) else 0
        print(f"{out_path}: {len(feats)} satır, {feats['track_id'].nunique()} iz "
              f"({n_dropped} kısa iz atlandı, {n_static} durağan)")
