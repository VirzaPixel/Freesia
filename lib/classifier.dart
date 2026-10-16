import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as pkg_image;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

/// Info satu spesies bunga (dari assets/flowers.json).
class FlowerInfo {
  FlowerInfo({
    required this.folder,
    required this.name,
    required this.english,
    required this.latin,
  });

  final String folder;
  final String name;
  final String english;
  final String latin;

  factory FlowerInfo.fromJson(Map<String, dynamic> j) => FlowerInfo(
        folder: j['folder'] as String,
        name: (j['name'] as String?) ?? j['folder'] as String,
        english: (j['en'] as String?) ?? '',
        latin: (j['latin'] as String?) ?? '',
      );
}

/// Satu kandidat hasil klasifikasi.
class Prediction {
  Prediction({required this.index, required this.score, required this.info});

  final int index;
  final double score;
  final FlowerInfo info;
}

/// Wrapper model/best.onnx (YOLO-cls → flutter_onnxruntime).
///
/// Fakta model:
/// - input 'images' NCHW float32 [1, 3, 640, 640], piksel diskala 1/255
/// - output 'output0' [1, 131] logit (tanpa softmax)
/// - urutan kelas = mapping indeks → class name (assets/class_names.json,
///   identik dengan metadata `names` bawaan best.onnx)
class FlowerClassifier {
  FlowerClassifier._(
    this._session,
    this.labels,
    this._infoByFolder,
  );

  static const String modelAsset = 'assets/best.onnx';
  static const String classNamesAsset = 'assets/class_names.json';
  static const int inputSize = 640;
  static const int numClasses = 131;

  final OrtSession _session;
  final List<String> labels;
  final Map<String, FlowerInfo> _infoByFolder;

  /// Parse JSON mapping `{"0": "01_Rose", ...}` → list terurut indeks numerik.
  static List<String> parseClassNames(String raw) {
    final map = jsonDecode(raw) as Map<String, dynamic>;
    final entries = map.entries.toList()
      ..sort((a, b) => int.parse(a.key).compareTo(int.parse(b.key)));
    return entries.map((e) => e.value as String).toList();
  }

  static Map<String, FlowerInfo> parseInfo(String raw) {
    final list = jsonDecode(raw) as List;
    return <String, FlowerInfo>{
      for (final e in list)
        (e as Map<String, dynamic>)['folder'] as String:
            FlowerInfo.fromJson(e),
    };
  }

  static Future<FlowerClassifier> load() async {
    final classNamesRaw = await rootBundle.loadString(classNamesAsset);
    final infoRaw = await rootBundle.loadString('assets/flowers.json');
    final labels = parseClassNames(classNamesRaw);
    if (labels.length != numClasses) {
      throw StateError(
          '$classNamesAsset harus $numClasses entri, ketemu ${labels.length}');
    }
    final infoByFolder = parseInfo(infoRaw);

    final ort = OnnxRuntime();
    final session = await ort.createSessionFromAsset(modelAsset);

    return FlowerClassifier._(session, labels, infoByFolder);
  }

  FlowerInfo infoFor(int index) {
    final folder = labels[index];
    return _infoByFolder[folder] ??
        FlowerInfo(folder: folder, name: folder, english: '', latin: '');
  }

  /// Klasifikasi bytes gambar (JPEG/PNG/WEBP apa pun) → topK prediksi.
  Future<List<Prediction>> classifyBytes(Uint8List bytes, int topK) async {
    final decoded = pkg_image.decodeImage(bytes);
    if (decoded == null) throw StateError('gambar tak terbaca');
    final oriented = pkg_image.bakeOrientation(decoded);
    // Scale sisi pendek → 640 lalu center-crop, setara preprocess training.
    final square = pkg_image.copyResizeCropSquare(oriented, size: inputSize);
    final inputFloatList = _toNchwFloat32(square);

    final inputOrt = await OrtValue.fromList(
      inputFloatList,
      [1, 3, inputSize, inputSize],
    );

    final inputName = _session.inputNames.isNotEmpty ? _session.inputNames[0] : 'images';
    final outputName = _session.outputNames.isNotEmpty ? _session.outputNames[0] : 'output0';
    final inputs = {inputName: inputOrt};

    final Map<String, OrtValue?> outputs;
    try {
      outputs = await _session.run(inputs);
    } finally {
      await inputOrt.dispose();
    }

    final outputOrt = outputs[outputName] ?? outputs.values.firstOrNull;

    if (outputOrt == null) {
      for (final val in outputs.values) {
        await val?.dispose();
      }
      throw StateError('Gagal mendapatkan output dari model ONNX');
    }

    final List<dynamic> rawVal;
    try {
      rawVal = await outputOrt.asList();
    } finally {
      for (final val in outputs.values) {
        await val?.dispose();
      }
    }
    List<double> logits;
    if (rawVal.isNotEmpty && rawVal.first is List) {
      logits = (rawVal.first as List).map((e) => (e as num).toDouble()).toList();
    } else {
      logits = rawVal.map((e) => (e as num).toDouble()).toList();
    }
    final probs = _softmax(logits);
    final order = List<int>.generate(probs.length, (i) => i)
      ..sort((a, b) => probs[b].compareTo(probs[a]));

    return order.take(topK.clamp(1, probs.length)).map((i) {
      final folder = labels[i];
      return Prediction(
        index: i,
        score: probs[i],
        info: _infoByFolder[folder] ??
            FlowerInfo(folder: folder, name: folder, english: '', latin: ''),
      );
    }).toList();
  }

  /// [square] 640×640 → Float32List NCHW [1,3,640,640], nilai 0..1.
  static Float32List _toNchwFloat32(pkg_image.Image square) {
    final out = Float32List(3 * inputSize * inputSize);
    final gOff = inputSize * inputSize;
    final bOff = 2 * inputSize * inputSize;
    var p = 0;
    for (var y = 0; y < inputSize; y++) {
      for (var x = 0; x < inputSize; x++) {
        final px = square.getPixel(x, y);
        out[p] = px.rNormalized.toDouble();
        out[gOff + p] = px.gNormalized.toDouble();
        out[bOff + p] = px.bNormalized.toDouble();
        p++;
      }
    }
    return out;
  }

  /// Diekspos untuk unit test (softmax wajib ter-normalisasi).
  static List<double> softmaxForTest(List<double> logits) =>
      _softmax(logits);

  static List<double> _softmax(List<double> logits) {
    var maxLogit = logits[0];
    for (final v in logits) {
      if (v > maxLogit) maxLogit = v;
    }
    final exps = List<double>.filled(logits.length, 0);
    var sum = 0.0;
    for (var i = 0; i < logits.length; i++) {
      final e = exp(logits[i] - maxLogit);
      exps[i] = e;
      sum += e;
    }
    for (var i = 0; i < exps.length; i++) {
      exps[i] /= sum;
    }
    return exps;
  }

  Future<void> close() async {
    await _session.close();
  }
}
