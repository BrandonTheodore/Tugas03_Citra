# Segmentasi Citra IF4073

Aplikasi berbasis MATLAB ini menampilkan segmentasi objek berbasis tepi, segmentasi region dengan Split and Merge, dan segmentasi semantik CamVid dengan LR-ASPP MobileNetV3-Large. Tab CamVid menjalankan pelatihan dan prediksi Python dari GUI yang sama.

## Dependensi dan cara menjalankan

Gunakan MATLAB R2022b atau lebih baru dengan Image Processing Toolbox. Tab CamVid memerlukan Python. Di Windows, GUI menggunakan Windows PowerShell untuk memulai proses Python. Pipeline CamVid diuji pada Python 3.12 dengan dependensi yang terletak di `requirements.txt`.

Siapkan lingkungan Python dari folder proyek. Contoh di Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Buka folder proyek di MATLAB, lalu run file `segmentation_app.m` atau jalankan perintah berikut.

```matlab
segmentation_app;
```

1. Klik **Buka citra** untuk memilih citra.
2. Pada **Deteksi Tepi**, pilih operator dan klik **Segmentasi objek**. GUI menampilkan citra asli, peta tepi, dan objek.
3. Pada **Split and Merge**, atur parameter lalu klik **Segmentasi region** untuk membandingkan hasil dengan citra asli.
4. Pada **Semantik CamVid**, gunakan Python dari `.venv` yang terdeteksi otomatis, atau pilih executable Python lain yang sudah memiliki dependensi. Gunakan checkpoint bawaan `output/camvid/camvid.pth`, lalu klik **Prediksi citra** untuk menampilkan hasil berwarna, overlay, dan legenda kelas.

## Anggota Kelompok

| NIM | Nama |
|---|---|
| 18223132 | Muhammad Rafly Fauzan     |
| 18223020 | Brandon Theodore Ferrinov |