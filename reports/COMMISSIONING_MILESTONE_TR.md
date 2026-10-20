# Dört maddede ne tamamlandı, nasıl gösterebilirsin?

20 Ekim 2026. Bu dosya öğrenci için kısa kullanım rehberidir. İngilizce akademik
metin [THESIS_DRAFT.md](../thesis/THESIS_DRAFT.md) dosyasındadır.

## 1. Artık sistem ne yapıyor?

Önce etiketli kaynak veriden “sağlıklı / arızalı” ayrımını öğreniyor. Sonra başka
bir çalışma koşulundaki titreşimleri, gerçek cevaplarına bakmadan kullanıyor.
Öğretmen modeli yüksek güvenli tahminler üretiyor. Öğrenci, sinyalin biraz
değiştirilmiş hâlinde bu tahmini öğrenmeye çalışıyor; aynı zamanda etiketli kaynak
örneklerinden öğrenmeye devam ediyor. Öğretmen, öğrencinin ağırlıklarının
hareketli ortalamasıyla güncelleniyor.

Bu işlem Paderborn için artık **uygulandı ve gerçekten çalıştırıldı**. Üç ayrı
başlangıç tohumu ve üç hedef koşul kullanıldı. Her hedef kendi kaynak modelinden
başlıyor. CWRU'dan Paderborn'a doğrudan aktarım yaptığımızı iddia etmiyoruz.

## 2. Hangi karşılaştırmaları yaptık?

| Yöntem | Basit açıklaması | Ortalama macro F1 |
|---|---|---:|
| İlk kaynak modeli | Öğren, yeni koşulda hiç değiştirmeden dene | %56,34 |
| Kaynakta ek eğitim | Yalnız eski etiketli verilerle biraz daha öğren | %65,42 |
| Kaynak EMA | Ek kaynak eğitiminin ağırlıklarını zaman içinde ortala | %66,84 |
| Öğretmen–öğrenci | Kaynak veriye ek olarak etiketsiz hedef veriden öğren | %66,38 |

Bu oranlar üç hedef koşul ve üç tohumdaki **kayıt düzeyi macro F1** ortalamasıdır.
“Doğruluk yüzdesi” veya “arızayı %66 olasılıkla buluyor” diye çevirmemelisin.

Çıkarımımız şu: ilk modele göre ilerleme var, ancak bu ilerlemenin büyük kısmı
sadece kaynakta ek eğitim ve ağırlık ortalamasıyla da elde ediliyor. Öğretmen–
öğrenci yöntemi kaynak EMA'yı dokuz karşılaştırmanın üçünde geçti, üçünde geride
kaldı, üçünde eşit kaldı. Ortalama fark -0,46 yüzde puan.

Bu, tez için saklanması gereken bir sonuç. “Yöntem her zaman daha iyi çalışır”
varsayımını deneyle kontrol etmiş olduk; sonuçları güzel göstermek için en iyi
tohumu seçmedik.

## 3. Nerede sorun var?

Düşük radyal kuvvette sistem çok fazla arızalı kaydı sağlıklı sanıyor. Öğretmenin
son eğitim turunda kabul ettiği sahte etiketlerin yalnızca %20,29–28,15'i doğru.
Buna rağmen kabul edilen tahminlerin ortalama güven skoru yaklaşık %98,50–99,16.
Yani modelin kendinden emin olması, haklı olduğu anlamına gelmiyor.

Öğrenci bu yanlış cevapları öğrenirse öğretmenle daha çok anlaşabilir. Bu yüzden
“iki model aynı cevabı veriyor” ölçüsü başarı kanıtı değildir. Eğitim bittikten
sonra, ayrı değerlendirme dosyalarındaki gerçek etiketlerle bu hataları saydık.
Bu gerçek etiketler eğitim veya model seçimine geri verilmedi.

Bazı koşullarda sıralama ölçüsü AUC çok yüksekken sabit 0,5 eşiği çok arıza
kaçırabiliyor. Bu, gelecekte karar eşiği/güven kalibrasyonu için araştırma nedeni;
mevcut test cevaplarına bakıp eşiği değiştirerek aynı deneyi “etiketsiz başarı”
diye sunamayız.

## 4. Bilgisayarda çalıştığını nasıl gösterirsin?

VS Code terminalini proje klasöründe aç:

```powershell
cd "C:\Users\TUF Gaming\Documents\Thesis"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Şu an 23 test geçiyor. Bu testler yazılımın belirli kurallarını kontrol eder;
sistemin endüstride güvenilir olduğu anlamına gelmez.

Uyarlanan öğretmenle sağlıklı bir kayıt:

```powershell
.\.venv\Scripts\python.exe scripts\predict_paderborn.py --file data\raw\paderborn\K006\N09_M07_F10_K006_1.mat --checkpoint outputs\paderborn_commissioning_v1\seed_7\N09_M07_F10\teacher_student.pt --output outputs\paderborn_commissioning_v1\demo\healthy.json
```

Bu ortamda ölçülen sonuç: `healthy`, arızalı sınıf skoru yaklaşık `0.0049`, 31
pencere, CUDA. Arızalı kayıt için:

```powershell
.\.venv\Scripts\python.exe scripts\predict_paderborn.py --file data\raw\paderborn\KA22\N09_M07_F10_KA22_1.mat --checkpoint outputs\paderborn_commissioning_v1\seed_7\N09_M07_F10\teacher_student.pt --output outputs\paderborn_commissioning_v1\demo\faulty.json
```

Ölçülen sonuç: `faulty`, skor yaklaşık `0.8903`, 31 pencere. İki örnek mevcut test
koleksiyonunun üyeleri; yeni ve bağımsız bir doğrulama deneyi değiller.

Tüm yeni raporları kontrol etmek için:

```powershell
.\.venv\Scripts\python.exe scripts\verify_commissioning_evidence.py
```

Eğitim komutunu mevcut tamamlanmış klasörde tekrar çalıştırmana gerek yok.
Program eski sonuçların üstüne yazmayı reddeder. Tekrarlanabilirlik kontrolü
zaten yapıldı: seçilen deneyin ağırlıkları, eğitim geçmişi ve sahte etiket izi
birebir eşleşti. GitHub'dan yeni indiren biri küçük modelleri
`scripts\restore_published_results.py` ile geri yükleyebilir; büyük veri dosyaları
ayrıca gereklidir.

## Okuma sırası

1. Önce bu dosyayı ve [sonuç raporundaki](PADERBORN_COMMISSIONING_RESULTS.md)
   ilk iki tabloyu oku.
2. [İngilizce tez taslağının](../thesis/THESIS_DRAFT.md) Özet, 1, 4, 6 ve 7.
   bölümlerini incele. Sonra veri ayrımını anlatan 3. bölüme dön.
3. `comparison_macro_f1.png`, `paired_deltas.png` ve
   `pseudo_label_diagnostics.png` grafiklerini açıklamayı dene.
4. Danışmanına “etiketsiz hedef veri, eşleştirilmiş kaynak EMA kontrolünün üzerine
   ne kadar katkı sağladı?” sorusuyla sonucu anlat.

Tamamlanan dört madde, çalışan bir prototip, karşılaştırmalı deney, hata analizi
ve ciddi bir yazılı taslak oluşturuyor. Tezin tam %60 olduğu veya teslim edilmeye
hazır olduğu iddia edilmiyor. Danışman geri bildirimi, bağımsız doğrulama,
seçilecek ek deneyler ve son yazım/düzenleme hâlâ gerekli. Bu aşamada yeni sunum
slaytı hazırlanmadı.
