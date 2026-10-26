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

## Takipçi karşılaştırması çekirdeğe alındı, ShanghaiTech ertelendi (2026-10-27)
Seçim: takipçi karşılaştırması stretch'ten çekirdeğe (Faz 7, MOT17'de IDF1/MOTA/HOTA).
ShanghaiTech indirmesi ertelendi (vakit kalırsa).
Neden: hoca ByteTrack'i sorguladı ve katman ayrımını istedi; karşılaştırma fiilen yapıldı ama
yalnız vekil ölçütlerle (Avenue'da takip etiketi yok). Seçimi sayısal kanıta bağlamak için
kimlik etiketli MOT17 şart. ShanghaiTech çok büyük; Ped1/Ped2/Avenue genelleme için şimdilik yeterli.

## MOT17: resmî zip, yalnız sabit kamera dizileri, kaynak = dizi klasörü (2026-10-27)
Seçim: motchallenge.net'ten MOT17.zip; train'den yalnız 02, 04, 09 (sabit kamera), FRCNN kopyası.
Kaynak olarak dizi klasörü verilir (`data/MOT17/train/MOT17-02-FRCNN`): kareler img1/'den,
fps seqinfo.ini'den okunur.
Neden: test etiketleri gizli (yalnız sunucuda), etiketli olan train; model eğitmediğimiz için
train'de ölçmek adil. Üç kopyanın (DPM/FRCNN/SDP) görüntü ve gt'si aynı, tespiti kendimiz yapıyoruz.
Roboflow sürümü tespit veri seti (ID yok, kareler karışık) → takip ölçümüne uygun değil.
Dizi klasörü: img1 verilse fps okunamaz (10 sanılır, MOT17 30) ve tüm diziler "img1" adını alırdı.

## Görsel çizim ölçüleri kare boyuyla orantılı (2026-10-27)
Seçim: çizgi kalınlığı, yazı boyu, nokta yarıçapı k = max(1, çıktı yüksekliği / 480) ile çarpılır.
Neden: ölçüler 480 px (Ped2) için sabitti; 1080 px MOT17'de kutular/ID'ler okunmuyordu.
max(1, ...): küçük karelerde (Ped, Avenue) çıktı birebir aynı kalır (Ped2'de 180/180 kare aynı).

## Avenue etiketleri: resmî piksel maskeleri → kare etiketi (2026-10-27)
Seçim: CUHK'nın resmî ground_truth_demo.zip'i (testing_label_mask/N_label.mat, volLabel: kare
başına 360x640 maske). Kare "anormal" = maskede en az bir anomali pikseli var. volLabel'ın i.
elemanı = videonun i. karesi (1'den sayım, frame_no ile aynı).
Neden: resmî kaynak; maskeden türetme literatürde yaygın. Bazı çalışmaların kullandığı hazır kare
etiket dosyalarıyla sayılar biraz farklı olabilir → raporda belirtilecek.
Not: 08 ve 18 tamamen, 17 neredeyse tamamen anormal → video başına AUROC tanımsız olur;
Faz 7'de tüm kareler birleştirilerek (micro) hesaplanacak.

## Öznitelikler: pandas, önce yumuşat sonra türev, hız iki birimde (2026-10-27)
Seçim: pandas (yeni bağımlılık); konum ve kutu yüksekliği 0.5 sn'lik, zamanda ortalanmış
pencereyle yumuşatılır; türev np.gradient(x, time_s) ile (eşit olmayan aralıklar → kapanma
boşlukları doğru); hız speed_px (piksel/sn) ve speed_h (boy/sn = kutu yüksekliğine bölünmüş);
yön 0–360° (0 sağ, 90 ekranda yukarı), hız < 0.1 boy/sn ise NaN; 10 kareden kısa izler atlanır.
Neden: türev titremeyi büyütür; pencere saniyede tanımlı → fps'ten bağımsız. Boy/sn perspektifi
giderir: yaya medyanı Ped2 7.6 px/sn vs MOT17-09 81 px/sn (10 kat), ama 0.25 vs 0.29 boy/sn.
Ped2 Test001'de bisiklet 1.01 boy/sn, yaya 0.25 → 4 kat ayrışıyor. Eşikler veriden seçilecek
(beklenen "yürüme ~0.8 boy/sn" ölçülen değerlerle tutmadı). Bölgeye mesafe Faz 3'e taşındı.

## İvme, yön değişimi, izin tek sınıfı (2026-10-27)
Seçim: ivme = yumuşatılmış hızın türevi, işaretli (+ hızlanma), accel_px ve accel_h; yön değişimi
= np.unwrap'lı açının türevi (°/sn, + saat yönünün tersi), duran kişide NaN; track_class = izin
en sık sınıfı (kare bazlı class da kalır).
Neden: ikinci türev titremeyi iki kez büyütür → hız da bir kez daha yumuşatılır. unwrap olmadan
359°→1° geçişi −358° sanılır (yapay daire testinde ölçülen 45.0 °/sn = gerçek). Sınıf iz içinde
değişebiliyor (Ped2 ID 153). Bilinen sınırlılıklar: (1) yön değişimi yavaşta gürültülü: |medyan|
0.1–0.15 boy/sn'de 46 °/sn, 0.5+ boy/sn'de 18 °/sn → kurallar onu yalnız yeterli hızda kullanmalı;
(2) Ped2 Test001'de bisiklet izi (139) 67–180 arası sağlam ama sürücü 4 kısa yaya izine bölünmüş
(153, 223, 268, 335) → araç anomalisini taşıyan iz aracın kendi izi; sürücünün izine güvenilmez.
