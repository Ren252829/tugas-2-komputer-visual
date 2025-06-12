# 🚦 Klasifikasi Objek Lalu Lintas menggunakan Simple CNN

Proyek ini menggunakan model Deep Learning sederhana (Simple CNN) untuk melakukan klasifikasi gambar objek lalu lintas seperti mobil, sepeda, bus, motor, dan orang. Dataset yang digunakan berasal dari Kaggle: [`yusufberksardoan/traffic-detection-project`](https://www.kaggle.com/datasets/yusufberksardoan/traffic-detection-project).

## 📁 Struktur Dataset
Dataset diunduh secara otomatis menggunakan `kagglehub`. Dataset mengikuti format anotasi YOLO:

```
train/
├── images/
├── labels/
```

Setiap gambar `.jpg` dalam folder `images/` memiliki file `.txt` yang sesuai di folder `labels/`, berisi ID kelas dan bounding box.

## 🔄 Alur Kerja

1. **Unduh Dataset**  
   Dataset diambil dari Kaggle dan diekstrak menggunakan `kagglehub`.

2. **Preprocessing Data**  
   - Gambar di-resize ke 128x128 piksel dan dinormalisasi.
   - Label dikonversi ke format one-hot encoding menggunakan `LabelBinarizer`.

3. **Pemetaan Label**
   ```
   {
     "0": "bicycle",
     "1": "bus",
     "2": "car",
     "3": "motorbike",
     "4": "person"
   }
   ```

4. **Arsitektur Model CNN**
   - 2 layer konvolusi dan max pooling
   - Fully connected layer dengan dropout
   - Output layer menggunakan softmax

5. **Pelatihan & Evaluasi**
   - Dataset dibagi menjadi 80% data latih dan 20% data uji
   - Model dilatih selama 10 epoch dengan batch size 32
   - Hasil evaluasi berupa akurasi dan classification report

6. **Visualisasi**
   - Menampilkan gambar dengan prediksi benar dan salah
   - Warna hijau untuk prediksi benar, merah untuk salah

## 📊 Hasil (Contoh)

- **Akurasi**: ~85-90% (tergantung epoch dan randomness)
- Performa tinggi untuk kelas mayoritas seperti `car` dan `person`
- Prediksi dapat ditingkatkan lebih lanjut dengan augmentasi data

## 🛠 Library yang Digunakan

| Library             | Kegunaan                                          |
|--------------------|----------------------------------------------------|
| `opencv-python`    | Membaca dan memproses gambar                       |
| `numpy`            | Operasi numerik                                    |
| `matplotlib`       | Visualisasi hasil                                  |
| `scikit-learn`     | Preprocessing label dan evaluasi model             |
| `tensorflow`       | Membangun dan melatih model CNN                    |
| `tqdm`             | Progress bar saat loading data                     |
| `kagglehub`        | Mengunduh dataset dari Kaggle                      |

## 📌 Cara Menjalankan

1. **Install dependency** (direkomendasikan dalam virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Jalankan file Python**:
   ```bash
   python tugas_klasifikasi2.py
   ```
   

## 📝 Penulis
Dibuat untuk tugas mata kuliah **Komputer Visual**, oleh Renaldi Pratama.
