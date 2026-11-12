# Bellman's Lost-in-a-Forest Problem: Deneysel Araştırma ve Doğrulama Günlüğü

Bu belge, Bellman'ın "Ormanda Kaybolma" problemi (1955) için Diferansiyel Evrim (`jDE`) ve Doğrusal Programlama (LP) tabanlı Chebyshev marjin testiyle elde edilen sayısal sonuçları, literatür karşılaştırmalarını ve teorik analizleri arşivler.

---

## 1. Kuramsal Çerçeve ve Konvansiyonlar

### 1.1 Problem Tanımı
Düzlemde bilinen bir $F \subset \mathbb{R}^2$ orman geometrisi için, başlangıç konumu ve yönelimi bilinmeyen bir yürüyüşçünün ormandan çıkmasını **en kötü durumda** garanti eden en kısa yol $\gamma$'nın uzunluğu $E(F)$ aranır:

$$E(F) = \inf \left\lbrace L \ge 0 : \forall g \in \mathcal{G}, \quad g(\gamma[0, L]) \not\subset F \right\rbrace$$

### 1.2 Simetri Grubu Konvansiyonu: $SE(2)$ vs $E(2)$ (Kritik Ayrım)
Literatürde iki farklı hareket grubu kullanılır; asimetrik geometrilerde bu ayrım esastır:
1. **Ötelemeli Dönmeler Grubu $SE(2)$ (Yalnızca Dönme + Öteleme / Oryantasyon Koruyan / Kiral):**
   - Orman $F$ dönebilir ve ötelenebilir, ancak **yansıtılamaz (aynalanamaz)**.
   - Yansıma simetrisi olmayan üçgenlerde (örn. 30-60-90, 3-4-5, Kepler, asimetrik üçgenler) kaçış mesafesi el yordamına (chirality) duyarlıdır.
   - Bir rota $SE(2)$ altında kaçışı sağlarken (marjin $\le 0$), ayna görüntüsü ormanın içine pozitif marjinle sığabilir.
2. **Tam Öklid Eşliği $E(2)$ (Dönme + Öteleme + Yansıma):**
   - Ormanın ve rotanın yansımalarına izin verilir. Klasik yay örtme literatürü (Wetzel 1973, Wichiramala & Panraksa 2026) bu konvansiyonu kullanır.
   - *Örnek:* 30-60-90 üçgeni için $SE(2)$ altında bulunan Z-rotası $E = \frac{3\sqrt{2}-\sqrt{6}}{2} \approx 0.896575$ iken, ayna görüntüsü üçgenin içine $+0.0104$ marjinle sığmaktadır. Wichiramala & Panraksa (2026) tam eşlik ($E(2)$) altında birim yayları örtmek için $E(\text{hip}=1) \ge \approx 0.911$ gerektiğini göstermiştir.

*Bu logdaki tüm asimetrik üçgen sonuçları aksi belirtilmedikçe **$SE(2)$ konvansiyonunda** elde edilmiştir.*

---

## 2. Literatür Karşılaştırması ve Bağımsız Doğrulama Özeti

| Geometri | Konvansiyon | Literatür Referansı | Motor Değeri $E(n)$ | Doğrulama Durumu |
|---|---|---|---|---|
| **Kare** (Kenar 1) | $E(2)$ | $\sqrt{2} \approx 1.414214$ (Bellman 1955) | `1.41422` | Doğrulama (Teorik optimum çaptır) |
| **Düzgün Beşgen** (Çap 1) | $E(2)$ | $1.000000$ (Klasik çap) | `1.000000` | Doğrulama ($k \ge 4$ düzgün çokgenlerde çap optimumdur) |
| **Eşkenar Üçgen** (Kenar 1) | $E(2)$ | $\sqrt{27/28} \approx 0.981981$ (Besicovitch 1965, Coulton–Movshovich 2006) | `0.981982` | Yeniden üretim ($+1.9 \times 10^{-6}$ hata, Z-rotası) |
| **İkizkenar Dik Üçgen** (Kenar 1) | $E(2)$ | $3/\sqrt{5} \approx 1.341641$ (Movshovich–Wetzel 2003) | `1.341641` | Yeniden üretim ($+2.5 \times 10^{-7}$ hata, $\theta = \arctan(3/4)$ Z-rotası) |
| **Altın Gnomon** (108-36-36) | $E(2)$ | $E \approx 0.79273$ (Temerev–Doria, Temmuz 2026; bacak=1 için $C=1.282676$) | `0.793052` | Bağımsız doğrulama (%0.04 analitik optimumun üstünde) |
| **Sonsuz Şerit** (Genişlik 1) | $E(2)$ | $\zeta \approx 2.278292$ (Zalgaller 1982 Broadworm) | `2.309401` ($n=8$, $4/\sqrt{3}$) | Poligonal yaklaşım |
| **30-60-90 Üçgeni** | **$SE(2)$** | $E = \frac{3\sqrt{2}-\sqrt{6}}{2} \approx 0.896575$ | `0.896576` | $SE(2)$ altında sınırda; $E(2)$ yansıma altında sığar |

---

## 3. İkizkenar Üçgenlerde Dal Kesişimi ($E_Z \to E_C$) ve jDE Dinamikleri

Tepe açısı $\alpha$ (taban açısı $\beta = (180^\circ - \alpha)/2$) ile parametrelendirilen ikizkenar üçgenlerde:
* **Analitik Literatür Zemin:**
  - Temerev & Doria (arXiv:2607.24483, Temmuz 2026) ve Gibbs (viXra) çalışmalarına göre:
    - Z-rotası dalı ile yay/tünel ("Tunnel" phase: iki dairesel omuz ve teğet doğru parçası) dalının analitik kesişimi **taban açısı $\beta \approx 42.29^\circ$ (tepe açısı $\alpha_c \approx 95.42^\circ$)** noktasında gerçekleşir.
    - Taban açısı $\beta \approx 27.6^\circ$ ($\alpha \approx 124.8^\circ$) civarında ise tünel eğrisi Zalgaller broadworm eğrisine birleşir.
* **Sayısal Optimizasyon ve jDE Dinamiği:**
  - Kaçış mesafesi $E(\alpha) = \min(E_Z(\alpha), E_C(\alpha))$ iki analitik dalın zarfıdır. Kesişimde türev süreksizliği ($kink$) jenerik bir dal kesişimidir; termodinamik bir faz geçişi değildir.
  - jDE stokastik popülasyonu $95.4^\circ \le \alpha \le 97.2^\circ$ aralığında yerel Z-havzasına takılabilmekte, $97.3^\circ$ civarında ise global C-havzasına atlamaktadır. $97.2^\circ-97.3^\circ$ civarındaki ani düşüş jDE'nin arama peyzajındaki havza geçişini yansıtır.

---

## 4. Şerit Teoremi ve Basık Üçgenler Sınırı ($\zeta h$ vs $4h/\sqrt{3}$)

Yüksekliği $h$ olan herhangi bir üçgen veya yamuk, genişliği $h$ olan sonsuz bir şeridin içine yerleştirilebilir.
Zalgaller (1982), genişliği $h$ olan bir şeritten kaçış için en kısa yolun broadworm eğrisi olduğunu ve uzunluğunun $\zeta h$ ($\zeta \approx 2.278292$) olduğunu kanıtlamıştır.
Dolayısıyla, **yüksekliği $h$ olan herhangi bir konveks orman için $\zeta h$ evrensel bir üst sınırdır**:

| Geometri | Yükseklik ($h$) | Zalgaller Sınırı $\zeta h$ | Motorun Bulduğu $E$ | Not |
|---|---|---|---|---|
| **`flat_wedge` (170°)** | $0.04373$ | **`0.09963`** | `0.100991` ($4h/\sqrt{3}$) | İki parçalı V-rotası şerit sınırının biraz üstünde kalır |
| **`asym_150` (150°)** | $0.11870$ | **`0.27044`** | `0.273817` | Kavisli J-kancası şerit sınırına çok yakındır |
| **`trapezoid`** | $0.40000$ | **`0.91132`** | `0.915430` | Poligonal kanca |
| **`obtuse_120` (120°)** | $0.28868$ | **`0.65769`** | `0.658263` ($n=14$) | Richardson ekstrapolasyonu ($0.6564$) şerit sınırının altındadır |

> **Analitik Formül Notu:** İki parçalı bir yol için şeritten çıkışın optimumu $60^\circ$ iç açılı eşkenar V-şeklidir ($E_{2\text{-seg}} = \frac{4h}{\sqrt{3}} \approx 2.3094 h$). Motorumuzun `flat_wedge` için bulduğu $0.100991$, $4h/\sqrt{3} \approx 0.100990$ değerini $10^{-6}$ hassasiyetle doğrular. Ancak Zalgaller'in kavisli rotası $\zeta h \approx 2.2783 h$ daha kısa bir üst sınır sağlar.

---

## 5. Deney Kayıtları ve Sayısal Sonuçlar

### 5.1 Kare ve Düzgün Çokgenler
* **Kare ($1 \times 1$, Çap $\sqrt{2}$):** $E = 1.41422$ (Bellman 1955 doğrulaması).
* **Düzgün Beşgen (Çap $1.0$):** $E = 1.000000$, tüm dönüşler $0.0^\circ$ (Tam doğrusal çap optimumdur).

### 5.2 Akut ve Dik Üçgenler (Z-Rotası Rejimi)
* **Eşkenar Üçgen (Kenar $1.0$):** $E = 0.981982$, Analitik Besicovitch: $\sqrt{27/28} \approx 0.981981$.
* **İkizkenar Dik Üçgen ($1, 1, \sqrt{2}$):** $E = 1.341641$, Analitik Movshovich–Wetzel: $3/\sqrt{5} \approx 1.341641$, Kırılmalar: $\pm 36.87^\circ$.
* **30-60-90 Üçgeni [$SE(2)$ Kiral]:** $E = 0.896576$, Analitik: $\frac{3\sqrt{2}-\sqrt{6}}{2} \approx 0.896575$, Kırılmalar: $\pm 60.00^\circ$. (Tam eşlik $E(2)$ altında $E \ge 0.911$).
* **3-4-5 Dik Üçgeni [$SE(2)$ Kiral]:** $E = 0.914905$, Kırılmalar: $+40.35^\circ, -44.79^\circ$.
* **Kepler Dik Üçgeni [$SE(2)$ Kiral]:** $E = 0.920663$, Kırılmalar: $\pm 45.9^\circ$.
* **Altın Üçgen (72-72-36):** $E = 0.986408$, Kırılmalar: $-15.70^\circ, +17.02^\circ, -16.54^\circ$.
* **Heron Üçgeni (13-14-15):** $E = 0.970143$, Kırılmalar: $+28.13^\circ, -28.11^\circ$.

### 5.3 Geniş Açılı Üçgenler (C/U-Kavisi Rejimi)
* **Altın Gnomon (108-36-36, Çap 1):** $E = 0.793052$. Temerev–Doria (2026) kesin çözümü $E \approx 0.79273$ ile uyumludur (+%0.04).
* **120-30-30 Üçgeni:** $E(14) = 0.658263$, Richardson limiti $E^* \approx 0.6564$, Dönüşler: pozitif tek yönlü kavis ($\approx 171.4^\circ$).
* **Asimetrik Geniş Açılı (`obtuse_asym`):** $E = 0.444033$, apeksi saran tek yönlü kanca.
* **20-60-100 Asimetrik Üçgen (`asym_100`):** $E = 0.660741$, tepe açısı $100^\circ > \alpha_c$ nedeniyle kanca formunda; $20^\circ-60^\circ$ asimetrisi nedeniyle $60^\circ$ kenarını saran deforme J-kancası.
* **10-20-150 Basık Üçgen (`asym_150`):** $E = 0.273817$, apeksi saran J-kancası.
* **5-5-170 Ultra-Basık Üçgen (`flat_wedge`):** $E = 0.100991$, eşkenar V-şekli ($120.0^\circ$ dış dönüş, $E = 4h/\sqrt{3}$).

### 5.4 Konkav Geometriler ve Dikkat Edilmesi Gereken Hususlar
* Konkav şekillerde (`l_shape`, `t_shape`, `plus_shape`), konveks kabuk (convex hull) testleri geçersizdir; rotanın parçalı birleşim içindeki kesin konumu test edilmelidir.
* L-şeklinde iç geodezik kiriş $\sqrt{1+\text{notch}^2}$ kaçış için yeterli bir uzunluktur ancak zorunlu minimum değildir (daha kısa kırık çizgiler köşeyi dolanabilir).
* `CompositePolygonForest` içindeki sığma testinin sürekli açı uzayında tam Lipschitz sertifikasyonu ile desteklenmesi gerekmektedir.

---

## 6. Yol Haritası ve Akademik Çerçeve

1. **Sertifikalı Doğrulama Paketi:**
   - Sayısal kayan nokta değerlerinin aralık aritmetiği (interval arithmetic) ile kesin üst sınırlara dönüştürülmesi.
   - Temerev–Doria (2026) süreklilik değerlerinin ($\alpha_c \approx 95.42^\circ$, $\beta \approx 27.6^\circ$) bağımsız doğrulanması.
2. **Kiralite ($SE(2)$) vs Tam Eşlik ($E(2)$) Karşılaştırması:**
   - İkizkenar olmayan üçgenlerde rotasyon-yalnız kaçış yolları ile ayna simetrisi altındaki örtme gereksinimlerinin sistematik haritalanması.
