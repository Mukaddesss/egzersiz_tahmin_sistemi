# aktivite_tahmin_sistemi

Bu proje, kullanıcıdan alınan yaş, cinsiyet, boy, kilo, ortalama nabız ve hedef kalori değeri ile en uygun aktivite planını oluşturan akıllı bir  sistemdir.

Model, XGBoost regresyonu kullanır ve her aktivite için dinamik süre + aktivite tahmini yaparak tekli veya çoklu antrenman bölümleri önerir. Ayrıca dataset fallback, nabız bölgesi hesaplama ve risk değerlendirmesi içerir.
