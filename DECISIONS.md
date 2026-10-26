# DECISIONS (karar günlüğü)

Her karar: **ne seçtik + neden + alternatif.** Savunmada "neden X?" sorularının cevabı burada.

## Sabit kamera varsayımı
Seçim: sabit güvenlik kamerası. Neden: piksel koordinatı = gerçek bölge, kamera hareketi
telafisi gerekmez, proje temiz kalır. Alternatif (İHA/hareketli kamera) = gelecek vizyonu.

## Konum = ayak noktası
Seçim: bbox alt-orta noktası. Neden: yerdeki bir bölgeyle nesnenin gerçek konumu en iyi
ayakla temsil edilir; merkez noktası yanıltır.

## Algı: YOLO + ByteTrack
Seçim: YOLO (hazır) + ByteTrack (gömülü, hafif, düşük-güvenli tespitleri de eşleştirir).
Neden: hızlı kurulum, gerçek-zamanlıya uygun; takip zorunlu (kimlik → yörünge → davranış).
Takipçi modüler: occlusion/ID-switch sorun olursa BoT-SORT (ReID) ya da karşılaştırma.
(2026-10-26 güncelleme: varsayılan takipçi OC-SORT oldu → "Tespit ve takip katmanları ayrıldı".)

## Karar katmanı: hibrit (kural + öğrenilmiş)
Seçim: kural kanalı + öğrenilmiş tek-sınıf kanalı, geç füzyon. Neden: saf kural kırılgan
(önceki projede görüldü), saf model kör alarm verir; ikisi birbirini dengeler ve
sistem açıklanabilir kalır.

## Öğrenilmiş model: önce Isolation Forest
Seçim: Isolation Forest (çekirdek), LSTM autoencoder (stretch). Neden: IF hafif, az veriyle
çalışır, açıklanabilir; "basit yeterli mi, derin ne kazandırır" karşılaştırması rapora değer
katar. Girdi = yörünge öznitelikleri (piksel değil) → hafif.

## Füzyon: geç karar füzyonu (IMADS-esintili)
Seçim: iki kanalın kararını ağırlıklı birleştir. Neden: kanal katkısı doğrudan açıklama olur
("İhlal %85 = kural 0.6 [oyalanma] + model 0.4 [anormal yörünge]").

## Veri kaydı: SQLite + SQLAlchemy ORM
Seçim: SQLite, SQLAlchemy ORM arkasında. Neden: yerel tek-kullanıcı için doğru (sunucusuz,
gömülü); ORM ile depolama soyut, gerekirse tek config ile PostgreSQL'e geçer. "SQLite basit"
itirazına karşı: bilinçli seçim + swappable mimari.

## Arayüz: PySide6
Seçim: native masaüstü (PySide6/Qt). Neden: akıcı video + etkileşimli bölge çizimi Qt'nin
güçlü yanı; yerel gözetim yazılımları zaten native masaüstüdür.

## Veri seti: UCSD Ped2 (önce)
Seçim: Ped2 (küçük, hızlı), sonra ShanghaiTech. Neden: Train = sadece normal (unsupervised'a
uygun), Test = anomali + hazır gt (etiketleme yok). Yaya sahnesi: odak insan, araç = anomali.

## Çalışma biçimi: offline-öncelik
Seçim: kayıtlı video üzerinde işle (değerlendirme için şart; canlı akış puanlanamaz).
Real-time = opsiyonel demo (sadece girdi kaynağını değiştir).

## Ortam: Python 3.11 + venv
Seçim: Python 3.11, proje içi .venv. Neden: torch/PySide6/ultralytics için olgun wheel desteği;
3.14 çok yeni (wheel eksikliği riski). venv ile proje paketleri sistemden izole.

## PyTorch ayrı kurulum (CUDA 12.8)
Seçim: torch requirements.txt dışında, pytorch.org cu128 index'inden önce kurulur. Neden:
Windows'ta pip varsayılanı CPU sürümü; RTX 5050 (Blackwell) CUDA ≥12.8 ister.

## Bağımlılıklar: minimum, faz faz
Seçim: yalnız ROADMAP'teki çekirdek paketler (+ ByteTrack için lap). pandas/matplotlib
ihtiyaç doğduğu fazda eklenecek. Neden: kapsam disiplini, gereksiz bağımlılık yok.

## Çoklu veri seti (hoca geri bildirimi, 2026-10-20)
Seçim: Ped2 ana geliştirme seti olarak kalır; Ped1, CUHK Avenue, ShanghaiTech (anomali) ve
MOT17-sabit kamera (takip) eklenir. Stretch'ten çekirdeğe alındı. Neden: tek veri seti
genellemeyi göstermez. Roller: Avenue → hızlı hareket + ters yön (Ped2'nin ölçemediği kural);
MOT17 → takip etiketli, takibi IDF1/MOTA ile sayısal ölçer; ShanghaiTech → çok sahne;
Ped1 → ucuz ek test. Sonuç: fps ve çözünürlük veri setine göre değişir → hız saniyede,
görselleştirme ölçeği ve fps kaynaktan okunur (sabit kodlanmaz).

## Görsel video: kaynağın gerçek fps'i
Seçim: çıktı video kaynağın fps'iyle yazılır; `--fps` yalnız sunum için hızlandırma.
Neden: sabit 10 fps, 30 fps'lik videoyu 3x ağır çekim yapıyordu (hoca: "oynatma yavaş").
Hızlandırılmış video gerçek hız gibi sunulmaz; çıktıda "x hızlandırılmış" yazar.

## Geliştirme ortamı: WSL2 Ubuntu 24.04 + Python 3.12 (2026-10-22)
Seçim: proje WSL2 içinde ayrı bir Ubuntu 24.04 dağıtımında, Linux diskinde (~/projects).
Neden: Windows Smart App Control imzasız DLL'leri (torch, SQLAlchemy) engelliyor, istisna listesi
yok ve bu bilgisayarda şu an kapatılması uygun değil. Ayrı dağıtım: ROS 2 kurulu 22.04 ortamından
izolasyon. Python 3.12: 24.04 varsayılanı, tüm paketler destekliyor (önceki 3.11 kararının yerine).
Veri /mnt/c yerine Linux diskinde: Windows diskinden okuma çok yavaş.

## Tespit ve takip katmanları ayrıldı (2026-10-26)
Seçim: `Detector.detect()` (N,6) dizi döndürür [x1,y1,x2,y2,güven,sınıf_no], conf=0.1; takipçi
ultralytics'ten doğrudan kurulur (`model.track()` yerine). Varsayılan OC-SORT, bekleme süresi
saniye cinsinden (3 sn → `track_buffer = sn × fps`).
Neden: hoca önerisi; her takipçi aynı tespitlerle çalışır → karşılaştırma adil. Sabit 30 kare
Avenue'da (25 fps) yalnız 1.2 sn ediyordu. OC-SORT Avenue 01+06+15'te ByteTrack'e göre ~%25 daha
az ID, %97 kapsama (geçici; MOT17 IDF1 ile kesinleşecek). Regresyon testi eski ölçümlerle birebir
aynı (ByteTrack 1.2 sn: 649 ID / 29623; OC-SORT 3 sn: 489 ID / 28799).

## Yörünge kaydı: CSV, takip bir kez yapılır (2026-10-26)
Seçim: her klip için `outputs/trajectories/<veriseti>_<klip>.csv`; sütunlar frame, time_s,
track_id, class, conf, ayak noktası (x_foot, y_foot), kutu (x1..y2).
Neden: takip yavaş (YOLO her karede), öznitelik/kural denemeleri hızlı olmalı → takip bir kez,
sonraki fazlar CSV'yi okur. time_s: farklı fps'li veri setlerinde hızı saniyede ölçmek için.
Kutu: hızı kişinin kameraya uzaklığına göre normalleştirmek ve sabit sahte kutuları ayıklamak için.
Dosya adında veri seti var (Test001 hem Ped1'de hem Ped2'de). CSV: ek paket yok, gözle kontrol edilir.
