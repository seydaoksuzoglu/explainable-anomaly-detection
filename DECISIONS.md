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
