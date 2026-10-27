"""
rules.py — Faz 3: kural kanalı (Kanal A). Öznitelik tablosundan gerekçeli olaylar üretir.

Kurallar (her biri tek satırlık olay + gerekçe cümlesi üretir):
  giris / bolgede_goruldu (3.3) : yasak (restricted) ya da oyalanma (loiter) bölgesine giriş
  hiz        (3.4a)             : sahne hız sınırının (speed_limit_h) üstünde hareket (koşma vb.)
  yon_ihlali (3.4a)             : yön bölgesinde (direction) izinli yönlerin dışında hareket
  oyalanma   (3.4b)             : oyalanma bölgesinde (loiter) kalış süresi bölgenin sınırını
                                  (max_dwell_s) aşıyor. Bu bölgeye GİRMEK serbest: giriş alarmı yok.

ORTAK MEKANİZMA — süreklilik (Persistence): ham koşul tek karede değil, KESİNTİSİZ bir süre
sağlanınca olay başlar; karşı koşul yine bir süre sağlanınca biter. Tek karelik kutu zıplamaları,
sınır titremesi ve yavaş harekette yönün gürültüsü böylece alarma dönüşmez (alarm yorgunluğu).
Bölge girişi bunun özel hali: başlama ve bitme koşulu farklı eşiklerde (histerezis).

İz uçları: features.py'de yumuşatma penceresi izin ilk/son ~0,25 sn'sinde tek taraflı kalır →
hız ve yön orada güvenilmez (DECISIONS). Hareket kuralları bu satırları atlar.

Kural dışı olaylar ve "bolgede_goruldu" için durağan izler (track_static) atlanır.

ÇIKTI (3.5): her klip için bir olay dosyası + bütün kliplerin olayları TEK tabloda
(<out-dir>/<sahne>_tum_olaylar.csv, "clip" sütunuyla). Bu tablo Kanal A'nın çıktısıdır;
Faz 5'teki füzyon ve Faz 6'daki olay günlüğü (SQLite) bunu okur. Bir satır = bir olay:
  clip, zone, zone_type, track_id, class, event, t_start, t_alarm, t_end, dwell_s,
  frame_start, frame_alarm, frame_end, peak, max_depth_h, closed, reason
  (peak: olayın en uç değeri — giriş/oyalanma: en derin nokta (boy), hız: en yüksek hız
   (boy/sn), yön: en büyük sapma (°); reason: Türkçe gerekçe cümlesi)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zones import load_scene_settings, load_zones, zone_features

# --- Bölge girişi (3.3) — Ped1 ölçümleriyle seçildi (DECISIONS.md) ---
ENTER_DEPTH_H = 0.15    # giriş için sınırdan en az bu kadar (boy) içeride olmalı
ENTER_SECONDS = 0.5     # ... ve bu kadar süre kesintisiz
EXIT_SECONDS = 1.0      # çıkış için bu kadar süre kesintisiz dışarıda

# --- Hareket kuralları (3.4a) ---
MOTION_SECONDS = 1.0    # hız koşulu en az bu kadar kesintisiz sürmeli
MOTION_OFF_SECONDS = 1.0
DIR_MIN_DIST_H = 0.2    # yön ihlali: yasak yönde KESİNTİSİZ en az bu kadar (boy) İLERLEMELİ.
                        # Süre değil yol: süre şartı hızlı geçeni kaçırıyordu (koşarak inen merdiveni
                        # 1 sn'den kısa sürede geçer); yerinde titreyen kutu ise yol biriktiremez.
                        # Avenue merdiven senaryosu: 0,2 boy = süre 0,5 sn kadar iniş yakalıyor
                        # (43/75), çıkanlara 0 sahte alarm (DECISIONS).
EDGE_SECONDS = 0.25     # izin ilk/son bu kadar süresi hız/yön için güvenilmez (tek taraflı pencere)
DIR_MIN_SPEED_H = 0.2   # yön ancak bu hızın üstünde anlamlı (Faz 2: yavaşta yön değişimi gürültülü)

ENTRY_ZONE_TYPES = ("restricted", "loiter")   # giriş/çıkış durum makinesi çalışan bölge türleri


class Persistence:
    """Genel süreklilik (debounce) durum makinesi: PASİF ↔ AKTİF.

    step(t, frame, on, off, value, amount):
      PASİF iken: `on` koşulu kesintisiz sağlanırken biriken miktar on_threshold'a ulaşırsa AKTİF'e
      geçer → olay AÇILIR. Miktar varsayılan olarak geçen SÜRE (sn); `amount` verilirse onun
      toplamı (yön kuralında katedilen YOL, boy).
      AKTİF iken: `off` koşulu kesintisiz off_seconds sürerse PASİF'e döner → olay KAPANIR.
      `on` ve `off` ayrı verilir: tümleyen olmak zorunda değil → histerezis kurulabilir
      (bölge girişinde: on = 0,15 boy içeride, off = dışarıda; aradaki şerit durumu değiştirmez).
      value: olay boyunca en büyüğü saklanır (en yüksek hız, en derin nokta, en büyük sapma).
    Gözlemleri zaman sırasıyla tek tek alır, geleceğe bakmaz → canlı videoda da kullanılabilir.
    Dönüş: açılan olay (dict, "closed" anahtarı yok) / kapanan olay (closed=True) / None.
    """

    def __init__(self, on_threshold, off_seconds):
        self.on_threshold, self.off_seconds = on_threshold, off_seconds
        self.active = False
        self.since = None          # karşı duruma geçiş adayı: koşulun başladığı (zaman, kare)
        self.acc = 0.0             # aday boyunca biriken miktar (süre ya da yol)
        self.prev_t = None
        self.event = None

    def step(self, t, frame, on, off, value=np.nan, amount=None):
        if not self.active:
            if on:
                if self.since is None:
                    self.since, self.acc = (t, frame), 0.0      # adayın ilk gözlemi: birikim 0
                else:
                    self.acc += (t - self.prev_t) if amount is None else amount
                self.prev_t = t
                if self.acc >= self.on_threshold - 1e-9:      # yeterince birikti → onay
                    self.active = True
                    self.event = {"t_start": self.since[0], "frame_start": self.since[1],
                                  "t_alarm": t, "frame_alarm": frame, "peak": value,
                                  "t_last": t, "frame_last": frame}
                    self.since = None
                    return dict(self.event)
            else:
                self.since = None                    # koşul bozuldu → aday sıfırlanır
            return None

        # AKTİF: olayı güncelle, bitme koşuluna bak
        if not np.isnan(value):
            self.event["peak"] = np.nanmax([self.event["peak"], value])
        self.event["t_last"], self.event["frame_last"] = t, frame
        if off:
            if self.since is None:
                self.since = (t, frame)
            if t - self.since[0] >= self.off_seconds - 1e-9:
                closed = dict(self.event, t_end=self.since[0], frame_end=self.since[1], closed=True)
                self.active, self.event, self.since = False, None, None
                return closed
        else:
            self.since = None
        return None

    def finish(self):
        """Gözlemler bittiğinde hâlâ aktifse açık olayı son gözlemde kapatır (closed=False)."""
        if self.active and self.event:
            return dict(self.event, t_end=self.event["t_last"], frame_end=self.event["frame_last"],
                        closed=False)
        return None


class ZoneState(Persistence):
    """TEK bir izin TEK bir bölgeye göre giriş durum makinesi (3.3) — Persistence'ın özel hali.

    on  = ayak noktası ≥ ENTER_DEPTH_H boy içeride (ENTER_SECONDS)
    off = ayak noktası bölgenin dışında (EXIT_SECONDS)
    Olay türü: iz daha önce dışarıda görüldüyse "giris", hiç görülmediyse "bolgede_goruldu".
    """

    def __init__(self):
        super().__init__(ENTER_SECONDS, EXIT_SECONDS)
        self.seen_outside = False

    def step(self, t, frame, depth):
        """depth = dist_<bölge> (boy; içeride eksi)."""
        if depth > 0 and not self.active:
            self.seen_outside = True
        e = super().step(t, frame, on=depth <= -ENTER_DEPTH_H, off=depth > 0, value=-depth)
        if e is not None and "closed" not in e:      # yeni açılan olay: türünü belirle
            self.event["event"] = e["event"] = "giris" if self.seen_outside else "bolgede_goruldu"
        if e is not None and e.get("closed"):
            self.seen_outside = True                 # dışarı çıktı: sonraki giriş gerçek giriş
        return e


# ------------------------------------------------------------------ gerekçe cümleleri
SINIF = {"person": "yaya", "bicycle": "bisiklet", "car": "araba", "motorcycle": "motosiklet",
         "truck": "kamyon", "skateboard": "kaykay"}


def num(v, k=1):
    return f"{v:.{k}f}".replace(".", ",")            # Türkçe ondalık: 12,5


def heading_word(h):
    """Yön açısı → okunur yön (0 sağ, 90 ekranda yukarı / kameradan uzağa, 180 sol, 270 aşağı / kameraya)."""
    words = ["sağa", "yukarı, kameradan uzağa", "sola", "aşağı, kameraya doğru"]
    return words[int(((h + 45) % 360) // 90)]


def reason(e):
    """Olay için okunur gerekçe cümlesi (açıklanabilirlik). Tablo/arayüz için → Türkçe."""
    kim = f"ID {e['track_id']} ({SINIF.get(e['class'], e['class'])})"
    sure = num(e["t_end"] - e["t_start"])
    son = "" if e["closed"] else " (olay sürerken iz bitti)"
    if e["event"] == "giris":
        return (f"{kim} '{e['zone']}' bölgesine {num(e['t_start'])} sn'de girdi, {sure} sn kaldı{son}; "
                f"sınırdan en fazla {num(e['peak'], 2)} boy içeri ilerledi.")
    if e["event"] == "bolgede_goruldu":
        return (f"{kim} iz başladığında zaten '{e['zone']}' bölgesindeydi ({num(e['t_start'])} sn), "
                f"{sure} sn kaldı{son}; sınırdan en fazla {num(e['peak'], 2)} boy içeri ilerledi.")
    if e["event"] == "hiz":
        return (f"{kim} {num(e['t_start'])} sn'den itibaren {sure} sn boyunca sahnenin normal hız "
                f"sınırının ({num(e['limit'], 2)} boy/sn) üstünde hareket etti{son}; "
                f"en yüksek hız {num(e['peak'], 2)} boy/sn.")
    if e["event"] == "oyalanma":
        nasil = (f"{num(e['t_start'])} sn'de girdi" if not e["started_inside"] else
                 f"iz başladığında zaten bölgedeydi ({num(e['t_start'])} sn; gerçek süre daha uzun olabilir)")
        return (f"{kim} '{e['zone']}' bölgesinde {sure} sn kaldı (sınır {num(e['limit'])} sn){son}; "
                f"{nasil}, alarm {num(e['t_alarm'])} sn'de.")
    if e["event"] == "yon_ihlali":
        izin = ", ".join(f"{a:.0f}°" for a in e["allowed"])
        return (f"{kim} '{e['zone']}' bölgesinde {num(e['t_start'])} sn'den itibaren {sure} sn boyunca "
                f"izinli yönlerin ({izin}) dışında, ~{e['heading']:.0f}° yönünde ({heading_word(e['heading'])}) "
                f"hareket etti{son}; en büyük sapma {e['peak']:.0f}°.")
    return f"{kim} {e['event']}"


def _run(sm, rows, *cond_fns):
    """Bir durum makinesini satırlar üzerinde çalıştırır → olay listesi (açık olaylar kapanmış haliyle)."""
    out = []
    for r in rows:
        e = sm.step(*cond_fns[0](r)) if cond_fns else None
        if e is None:
            continue
        if "closed" in e:
            out[-1] = e                              # açık olayın kapanmış hali
        else:
            out.append(e)
    last = sm.finish()
    if last is not None:
        out[-1] = last
    return out


# ------------------------------------------------------------------ kurallar
def zone_events(df, zones):
    """Bölge olayları. Her iz ve bölge için 3.3'ün giriş/çıkış durum makinesi çalışır:
      restricted: her onaylı giriş bir olay ("giris" / "bolgede_goruldu").
      loiter    : giriş serbest; bölgede kalış (olayın süresi) z.max_dwell_s'i aşarsa "oyalanma".
                  Alarm anı = giriş anı + sınır (operatör o anda uyarılır). Kalış süresi debounce'lu
                  giriş/çıkıştan geldiği için sınırda titreyen kişide süre sıfırlanmaz.
    """
    zones = [z for z in zones if z.type in ENTRY_ZONE_TYPES]
    events = []
    for tid, g in df.sort_values("frame").groupby("track_id", sort=False):
        static, cls = bool(g["track_static"].iloc[0]), g["track_class"].iloc[0]
        rows = list(zip(g["time_s"], g["frame"]))
        for z in zones:
            depths = g[f"dist_{z.name}"].to_numpy()
            sm = ZoneState()
            evs = _run(sm, range(len(g)), lambda i: (rows[i][0], rows[i][1], depths[i]))
            for e in evs:
                if e["event"] == "bolgede_goruldu" and static:
                    continue                         # durağan iz içeride başlıyorsa: sahte nesne
                e.update(zone=z.name, zone_type=z.type, track_id=int(tid), **{"class": cls})
                e["max_depth_h"] = e["peak"]
                if z.type == "loiter":
                    e = _loiter(e, g, z)             # giriş değil kalış süresi önemli
                    if e is None:
                        continue
                events.append(e)
    return events


def _loiter(e, g, z):
    """Bir oyalanma bölgesi olayını (giriş → çıkış) kalış sınırına göre "oyalanma"ya çevirir.
    Kalış sınırı aşılmadıysa None (bu bölgeye girmek serbest)."""
    if e["t_end"] - e["t_start"] < z.max_dwell_s - 1e-9:
        return None
    t_alarm = e["t_start"] + z.max_dwell_s
    after = g[g["time_s"] >= t_alarm - 1e-9]         # alarm anına denk gelen ilk gözlem
    return dict(e, event="oyalanma", started_inside=e["event"] == "bolgede_goruldu",
                limit=z.max_dwell_s, t_alarm=float(after["time_s"].iloc[0]),
                frame_alarm=int(after["frame"].iloc[0]))


def _usable(g):
    """Hız/yön için güvenilir satırlar: izin ilk/son EDGE_SECONDS'ı hariç."""
    t = g["time_s"]
    return (t - t.iloc[0] >= EDGE_SECONDS) & (t.iloc[-1] - t >= EDGE_SECONDS)


def speed_events(df, speed_limit_h):
    """Hız kuralı (3.4a): speed_h ≥ sınır, kesintisiz MOTION_SECONDS."""
    events = []
    for tid, g in df.sort_values("frame").groupby("track_id", sort=False):
        if bool(g["track_static"].iloc[0]):
            continue
        g = g[_usable(g)]
        if g.empty:
            continue
        t, fr, v = g["time_s"].to_numpy(), g["frame"].to_numpy(), g["speed_h"].to_numpy()
        sm = Persistence(MOTION_SECONDS, MOTION_OFF_SECONDS)
        for e in _run(sm, range(len(g)), lambda i: (t[i], fr[i], v[i] >= speed_limit_h,
                                                       v[i] < speed_limit_h, v[i])):
            e.update(event="hiz", zone="-", zone_type="-", track_id=int(tid),
                     limit=speed_limit_h, **{"class": g["track_class"].iloc[0]})
            events.append(e)
    return events


def angular_gap(h, allowed):
    """Yön h'nin izinli yönlerin EN YAKININA açısal uzaklığı (0-180°). 359° ile 1° arası 2°."""
    d = np.abs((np.asarray(h)[:, None] - np.asarray(allowed)[None, :] + 180) % 360 - 180)
    return d.min(axis=1)


def direction_events(df, zones):
    """Yön kuralı (3.4a): yön bölgesinin içinde, yeterince hızlı ve izinli yönlerden
    heading_tolerance'tan fazla sapan hareket; yasak yönde kesintisiz DIR_MIN_DIST_H boy ilerleme."""
    zones = [z for z in zones if z.type == "direction"]
    events = []
    for tid, g in df.sort_values("frame").groupby("track_id", sort=False):
        if bool(g["track_static"].iloc[0]):
            continue
        g = g[_usable(g)]
        if g.empty:
            continue
        t, fr = g["time_s"].to_numpy(), g["frame"].to_numpy()
        # her gözlemde katedilen yol (boy): hız × önceki gözlemden geçen süre
        step_h = g["speed_h"].to_numpy() * np.diff(t, prepend=t[0])
        moving = (g["speed_h"] >= DIR_MIN_SPEED_H).to_numpy() & g["heading"].notna().to_numpy()
        for z in zones:
            gap = angular_gap(g["heading"].fillna(0).to_numpy(), z.allowed_headings)
            bad = moving & g[f"in_{z.name}"].to_numpy(bool) & (gap > z.heading_tolerance)
            sm = Persistence(DIR_MIN_DIST_H, MOTION_OFF_SECONDS)       # eşik: YOL (boy)
            for e in _run(sm, range(len(g)), lambda i: (t[i], fr[i], bad[i], not bad[i],
                                                           gap[i] if bad[i] else np.nan, step_h[i])):
                span = g[(g["frame"] >= e["frame_start"]) & (g["frame"] <= e["frame_end"])]
                # olay boyunca tipik yön: açıların dairesel ortalaması (359° ve 1° → 0°)
                r = np.radians(span["heading"].dropna())
                e.update(event="yon_ihlali", zone=z.name, zone_type=z.type, track_id=int(tid),
                         allowed=z.allowed_headings,
                         heading=float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360),
                         **{"class": g["track_class"].iloc[0]})
                events.append(e)
    return events


COLUMNS = ["zone", "zone_type", "track_id", "class", "event", "t_start", "t_alarm", "t_end",
           "dwell_s", "frame_start", "frame_alarm", "frame_end", "peak", "max_depth_h", "closed",
           "reason"]


REQUIRED = ("track_static", "track_class", "speed_h", "heading", "h_s")   # features.py (3.2+) sütunları


def all_events(df, zones, speed_limit_h=None):
    """Bütün kuralların olaylarını tek tabloda toplar."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Öznitelik dosyasında {missing} sütunları yok: dosya eski bir features.py "
                         "sürümüyle üretilmiş. features.py ile yeniden üret.")
    if any(f"dist_{z.name}" not in df.columns for z in zones):
        df = zone_features(df, zones)                # bölge sütunları yoksa ekle
    events = zone_events(df, zones) + direction_events(df, zones)
    if speed_limit_h is not None:
        events += speed_events(df, speed_limit_h)
    for e in events:
        e["dwell_s"] = e["t_end"] - e["t_start"]
        e["reason"] = reason(e)
    return pd.DataFrame(events, columns=COLUMNS).sort_values(["t_alarm", "track_id"]).reset_index(drop=True)


def calibrate(paths, q=0.99):
    """Hız sınırı önerisi: YALNIZ NORMAL davranış içeren (eğitim) videolarda, hareket eden
    kişilerin speed_h dağılımının q yüzdeliği. Test verisine bakılmaz → veri sızıntısı yok."""
    vals = []
    for p in paths:
        df = pd.read_csv(p)
        for _, g in df.groupby("track_id"):
            if bool(g["track_static"].iloc[0]) or g["track_class"].iloc[0] != "person":
                continue
            g = g[_usable(g)]
            vals.append(g["speed_h"].to_numpy())
    v = np.concatenate(vals)
    print(f"{len(paths)} dosya, {len(v)} satır (yaya, durağan ve iz uçları hariç)")
    for qq in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999):
        print(f"  %{qq * 100:g} yüzdelik: {np.quantile(v, qq):.3f} boy/sn")
    lim = float(np.quantile(v, q))
    print(f'Öneri → sahne dosyasına ekle:  "speed_limit_h": {lim:.2f}')
    return lim


# Kullanım:
#   Eşik önerisi (yalnız normal/eğitim videoları):
#     python src/rules.py --calibrate outputs/features/Avenue_train_*.csv
#   Olaylar:
#     python src/rules.py outputs/features/UCSDped1_Test0*.csv --zones configs/zones/UCSDped1.json
#     python src/rules.py outputs/features/Avenue_??.csv --zones configs/zones/Avenue.json
#   Oyalanma bölgesi: zones.py draw ... --type loiter --max-dwell 10
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Öznitelik tablolarından kural olayları üretir.")
    parser.add_argument("features", nargs="+", help="öznitelik CSV'leri (features.py çıktısı)")
    parser.add_argument("--zones", help="sahne dosyası (configs/zones/<veriseti>.json)")
    parser.add_argument("--speed-limit", type=float,
                        help="hız sınırı (boy/sn); verilmezse sahne dosyasındaki speed_limit_h")
    parser.add_argument("--calibrate", action="store_true",
                        help="olay üretme; verilen (NORMAL) videolardan hız sınırı öner")
    parser.add_argument("--out-dir", default="outputs/events")
    args = parser.parse_args()

    if args.calibrate:
        calibrate(args.features)
        raise SystemExit
    if not args.zones:
        parser.error("--zones gerekli (olay üretirken)")

    zones, _ = load_zones(args.zones)
    limit = args.speed_limit if args.speed_limit is not None else \
        load_scene_settings(args.zones).get("speed_limit_h")
    if limit is None:
        print("Not: hız sınırı yok (speed_limit_h / --speed-limit) → hız kuralı çalışmıyor.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, tot_frames, all_ev = {}, 0, []
    entry_zones = [z for z in zones if z.type == "restricted"]   # kare bazlı karşılaştırma: yasak bölge
    for path in args.features:
        df = pd.read_csv(path)
        if df.empty:
            continue
        ev = all_events(df, zones, limit)
        ev.to_csv(out_dir / Path(path).name, index=False, float_format="%.3f")
        all_ev.append(ev.assign(clip=Path(path).stem))     # tek tablo için: hangi klipten
        # Karşılaştırma (raporun İP7'si): kare bazlı yaklaşım, giriş bölgelerinde her "içeride"
        # karesinde alarm verirdi (durağan izler hariç, adil olsun).
        moving = df[~df["track_static"].astype(bool)]
        if entry_zones:
            zf = zone_features(moving, entry_zones)
            tot_frames += int(sum(zf[f"in_{z.name}"].sum() for z in entry_zones))
        for k, n in ev["event"].value_counts().items():
            counts[k] = counts.get(k, 0) + n
        print(f"{Path(path).stem}: {len(ev)} olay")
        for r in ev["reason"]:
            print("   ", r)
    n_entry = counts.get("giris", 0) + counts.get("bolgede_goruldu", 0)
    print("TOPLAM:", counts or "olay yok")

    # --- 3.5: bütün kliplerin olayları tek tabloda + özet ---
    if all_ev:
        table = pd.concat(all_ev, ignore_index=True)[["clip"] + COLUMNS]
        all_path = out_dir / f"{Path(args.zones).stem}_tum_olaylar.csv"
        table.to_csv(all_path, index=False, float_format="%.3f")
        print(f"\nTek tablo: {all_path} ({len(table)} olay, {table['clip'].nunique()} klip)")
        if len(table):
            # kural × bölge özeti: kaç olay, kaç farklı klipte, ortalama süre
            ozet = table.assign(zone=table["zone"].replace("-", "(sahne)")).groupby(["event", "zone"]).agg(
                olay=("track_id", "size"), klip=("clip", "nunique"), ort_sure_sn=("dwell_s", "mean"))
            print(ozet.round(1).to_string())
    if entry_zones:
        print(f"Bölge girişi: kare bazlı {tot_frames} alarm → durum geçişi {n_entry} olay")
