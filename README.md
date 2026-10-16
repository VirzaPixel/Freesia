# Freesia — Deteksi Bunga dengan AI (Flutter + ONNX)

Aplikasi Android berbasis Flutter untuk mengenali **131 spesies bunga** secara *offline*
menggunakan model klasifikasi gambar **YOLO-cls (`best.onnx`)** yang dijalankan lewat
**ONNX Runtime** (`flutter_onnxruntime`).

Semua inferensi berjalan **100% di perangkat** — tidak ada gambar atau data pengguna
yang dikirim ke server mana pun.

---

## Fitur

- Klasifikasi 131 spesies bunga (nama Indonesia, nama Inggris, nama Latin).
- Dua sumber gambar: **Kamera** dan **Galeri**.
- UI Material 3 modern: tema terang/gelap, katalog spesies dengan pencarian,
  indikator loading multi-tahap (parsing gambar → inferensi ONNX).
- Hasil prediksi: persentase confidence 2 desimal, class name bawaan model
  (mis. `12_Lily`), waktu inferensi (ms), top-5 kandidat.
- Berjalan tanpa koneksi internet (offline-first).

## Arsitektur

```
lib/
├── main.dart        # UI Material 3 (theme, HomePage, katalog spesies)
└── classifier.dart  # Wrapper ONNX: decode → resize/crop 640×640 → NCHW
                     # → run session → softmax → top-K

assets/
├── best.onnx        # Model YOLO-cls (215 MB) — TIDAK di-commit, lihat di bawah
├── class_names.json # Mapping indeks kelas → class name (131 entri)
└── flowers.json     # Metadata spesies: nama ID/EN, nama Latin
```

### Alur inferensi

1. `ImagePicker` memilih gambar (maks. 1280 px, kualitas 92).
2. `pkg_image.decodeImage` + `bakeOrientation` (normalisasi EXIF).
3. `copyResizeCropSquare` ke 640×640 (setara preprocessing training).
4. Konversi ke tensor NCHW float32 `[1,3,640,640]`, skala `1/255`.
5. `OrtSession.run` → logit `[1,131]` → `softmax` → urutkan → top-K.
6. Semua `OrtValue` di-`dispose()` di blok `finally` (tanpa kebocoran memori native).

## Persyaratan

- Flutter SDK (Dart `^3.12.2`)
- Android SDK (build via Android Studio / `flutter build apk`)
- File model `assets/best.onnx` (lihat bagian Model)

## Menyiapkan Model

`best.onnx` **tidak ikut di-commit** ke repository karena ukurannya ±215 MB
(melampaui batas GitHub). Letakkan file model secara manual sebelum build:

```bash
# Salin model dari sumber penyimpanan kamu ke:
cp /path/ke/best.onnx assets/best.onnx
```

File model dibutuhkan saat build — tanpa file ini, `flutter build` gagal
karena asset `assets/best.onnx` tidak ditemukan.

## Menjalankan

```bash
flutter pub get
flutter run          # jalankan di perangkat/emulator
```

Build rilis:

```bash
flutter build apk --release
```

## Pengujian & Kualitas

```bash
flutter analyze      # linter (flutter_lints)
flutter test         # unit test + widget test
```

Cakupan pengujian saat ini:
- `softmax` ternormalisasi dan monoton.
- `parseClassNames` mengurutkan indeks numerik dari JSON.
- `parseInfo` membaca metadata spesies.
- Format confidence (2 desimal).
- Render dasar UI aplikasi.

## Keamanan

- **Offline penuh**: tidak ada izin `INTERNET` di manifest rilis (`main`);
  izin `INTERNET` hanya ditambahkan Flutter di varian `debug`/`profile`
  untuk kebutuhan tooling.
- **Satu-satunya izin rilis**: `CAMERA` (wajib untuk fitur ambil foto;
  ditandai `android.hardware.camera.any` dengan `required="false"`).
- **Tidak ada data keluar**: gambar yang dipilih hanya dibaca ke memori lokal
  (`readAsBytes`) dan langsung diklasifikasi di perangkat.
- **Tanpa kredensial**: tidak ada API key, token, atau rahasia di dalam kode
  maupun konfigurasi build.
- **Rilis aman**: `android/app/build.gradle.kts` hanya memakai signing debug
  sebagai default — untuk distribusi, siapkan konfigurasi signing rilis kamu
  sendiri (jangan commit keystore/key.properties ke repository).
- File kebesaran/berisiko (model, keystore, build output) masuk `.gitignore`.

## Lisensi Model

Model `best.onnx` diekspor dengan Ultralytics (AGPL-3.0).
Perhatikan ketentuan lisensi AGPL-3.0 jika kamu mendistribusikan aplikasi ini.
