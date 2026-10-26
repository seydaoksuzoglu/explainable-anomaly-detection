"""
features.py — Faz 2: yörüngeden davranış öznitelikleri çıkarmak.

Girdi : trajectory.py'nin yazdığı yörünge CSV'si (bir satır = bir nesnenin bir karedeki konumu).
Çıktı : aynı satırlar + öznitelik sütunları. Kural kanalı (Faz 3) ve Isolation Forest (Faz 4)
        bu tabloyu okur; videoya bir daha dokunmaz.

Adım 2.1: yumuşatma, hız (piksel/sn ve boy/sn), yön, kalış süresi.
Adım 2.2: ivme, yön değişimi, izin tek sınıfı (track_class).

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


def _track_features(g):
    """TEK bir izin satırlarına öznitelik sütunlarını ekler (g = o izin tablosu)."""
    g = g.copy()
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
    return g


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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yörünge CSV'lerinden öznitelik tablosu üretir.")
    parser.add_argument("trajectory", nargs="+", help="bir veya birden çok yörünge CSV'si")
    parser.add_argument("--out-dir", default="outputs/features",
                        help="öznitelik CSV'lerinin yazılacağı klasör")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.trajectory:
        df = load_trajectory(path)
        feats = compute_features(df)
        out_path = out_dir / Path(path).name     # aynı ad: UCSDped2_Test001.csv
        # float_format: ondalıkları 3 basamağa kısalt (dosya küçük, gözle okunur kalsın)
        feats.to_csv(out_path, index=False, float_format="%.3f")
        n_dropped = df["track_id"].nunique() - feats["track_id"].nunique()
        print(f"{out_path}: {len(feats)} satır, {feats['track_id'].nunique()} iz "
              f"({n_dropped} kısa iz atlandı)")
