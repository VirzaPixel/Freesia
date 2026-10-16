import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'classifier.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const FreesiaApp());
}

class FreesiaApp extends StatelessWidget {
  const FreesiaApp({super.key});

  @override
  Widget build(BuildContext context) {
    const primaryColor = Color(0xFF1E5E3A); // Deep botanical emerald
    const secondaryColor = Color(0xFFE88E38); // Warm floral amber

    return MaterialApp(
      title: 'Freesia — Botanical AI',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.system,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.light,
        colorScheme: ColorScheme.fromSeed(
          seedColor: primaryColor,
          brightness: Brightness.light,
          primary: primaryColor,
          secondary: secondaryColor,
          surface: const Color(0xFFF9FBF9),
        ),
        scaffoldBackgroundColor: const Color(0xFFF4F7F4),
        cardTheme: CardThemeData(
          elevation: 0,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
            side: const BorderSide(color: Color(0xFFE6EFE6), width: 1.2),
          ),
        ),
        fontFamily: null,
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF4CAF79),
          brightness: Brightness.dark,
          primary: const Color(0xFF62D290),
          secondary: const Color(0xFFFFB067),
          surface: const Color(0xFF151D18),
        ),
        scaffoldBackgroundColor: const Color(0xFF0F1511),
        cardTheme: CardThemeData(
          elevation: 0,
          color: const Color(0xFF1B241F),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
            side: const BorderSide(color: Color(0xFF24332A), width: 1.2),
          ),
        ),
      ),
      home: const HomePage(),
    );
  }
}

enum LoadState { loading, ready, parsing, inferencing, error }

extension LoadStateX on LoadState {
  bool get isProcessing => this == LoadState.parsing || this == LoadState.inferencing;
}

/// Format skor probabilitas menjadi persentase dengan 2 desimal presisi (misal: 97.79%, 0.62%, 0.13%).
String formatConfidence(double score) {
  final pct = score * 100.0;
  return pct.toStringAsFixed(2);
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _picker = ImagePicker();
  FlowerClassifier? _clf;
  String? _error;
  LoadState _state = LoadState.loading;

  Uint8List? _photo;
  List<Prediction> _results = const [];
  int _inferMs = 0;
  String _source = '';
  String _parsePhase = '';

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _clf?.close();
    super.dispose();
  }

  Future<void> _loadModel() async {
    setState(() {
      _state = LoadState.loading;
      _error = null;
    });
    try {
      final clf = await FlowerClassifier.load();
      if (!mounted) {
        clf.close();
        return;
      }
      setState(() {
        _clf = clf;
        _state = LoadState.ready;
      });
    } catch (e, stack) {
      debugPrint('Error loading ONNX model: $e\n$stack');
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _state = LoadState.error;
      });
    }
  }

  Future<void> _pick(ImageSource source, String label) async {
    final clf = _clf;
    if (clf == null || _state.isProcessing) return;

    try {
      final file = await _picker.pickImage(
        source: source,
        maxWidth: 1280,
        maxHeight: 1280,
        imageQuality: 92,
      );
      if (file == null) return;

      // Fase 1: Membaca file dan parsing gambar (langsung tampilkan preview/loading)
      setState(() {
        _state = LoadState.parsing;
        _parsePhase = 'Membaca dan memuat file gambar...';
        _source = label;
        _error = null;
        _results = const [];
      });

      final bytes = await file.readAsBytes();
      if (!mounted) return;

      setState(() {
        _photo = bytes;
        _parsePhase = 'Decoding dan normalisasi piksel (640×640)...';
      });

      // Waktu singkat agar transisi UI dan animasi loading parsing terlihat jelas oleh pengguna
      await Future<void>.delayed(const Duration(milliseconds: 350));
      if (!mounted) return;

      // Fase 2: Inferensi neural network ONNX
      setState(() {
        _state = LoadState.inferencing;
        _parsePhase = 'Menganalisis karakteristik bunga dengan model ONNX...';
      });

      final sw = Stopwatch()..start();
      try {
        final preds = await clf.classifyBytes(bytes, 5);
        sw.stop();
        if (!mounted) return;
        setState(() {
          _results = preds;
          _inferMs = sw.elapsedMilliseconds;
          _state = LoadState.ready;
          _parsePhase = '';
        });
      } catch (e, stack) {
        sw.stop();
        debugPrint('Inference error: $e\n$stack');
        if (!mounted) return;
        setState(() {
          _error = 'Gagal menganalisis gambar: $e';
          _state = LoadState.error;
          _parsePhase = '';
        });
        _showToast(context, 'Gagal memproses gambar: $e', isError: true);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _state = LoadState.ready;
        _parsePhase = '';
      });
      _showToast(context, 'Tidak dapat mengakses $label: $e', isError: true);
    }
  }

  void _clearImage() {
    setState(() {
      _photo = null;
      _results = const [];
      _inferMs = 0;
      _source = '';
      _error = null;
    });
  }

  void _showToast(BuildContext context, String message, {bool isError = false}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            Icon(
              isError ? Icons.error_outline_rounded : Icons.check_circle_outline_rounded,
              color: Colors.white,
              size: 20,
            ),
            const SizedBox(width: 10),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: isError ? Colors.red.shade700 : const Color(0xFF1E5E3A),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        margin: const EdgeInsets.all(16),
      ),
    );
  }

  void _showSpeciesCatalog() {
    final clf = _clf;
    if (clf == null) return;

    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => _SpeciesCatalogSheet(classifier: clf),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final busy = _state.isProcessing;
    final canPick = _clf != null && !busy && _state != LoadState.loading;

    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: colorScheme.primary.withValues(alpha: 0.12),
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.local_florist_rounded,
                color: colorScheme.primary,
                size: 22,
              ),
            ),
            const SizedBox(width: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Freesia',
                  style: TextStyle(fontWeight: FontWeight.w800, fontSize: 19, letterSpacing: -0.3),
                ),
                Text(
                  'Botanical Vision AI',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                    color: colorScheme.onSurface.withValues(alpha: 0.6),
                  ),
                ),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Katalog Spesies',
            icon: const Icon(Icons.menu_book_rounded),
            onPressed: _clf != null ? _showSpeciesCatalog : null,
          ),
          const SizedBox(width: 8),
        ],
        centerTitle: false,
        backgroundColor: theme.scaffoldBackgroundColor,
        elevation: 0,
        scrolledUnderElevation: 2,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
          children: [
            _buildStatusHeader(context),
            const SizedBox(height: 18),
            _buildActionButtons(context, canPick: canPick),
            const SizedBox(height: 18),
            if (_photo != null) _buildPhotoCard(context, busy: busy),
            if (_photo == null && busy) _buildParsingPlaceholder(context),
            if (busy) _buildAnalyzingState(context),
            if (!busy && _results.isNotEmpty) ...[
              const SizedBox(height: 20),
              _buildTopResultCard(context, _results.first),
              const SizedBox(height: 16),
              _buildAlternativeHeader(context),
              const SizedBox(height: 8),
              ..._results.skip(1).map((p) => _buildPredictionTile(context, p)),
            ],
            if (_photo == null && !busy && _state == LoadState.ready)
              _buildEmptyState(context),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusHeader(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    Color badgeBg;
    Color badgeFg;
    IconData badgeIcon;
    String title;
    String subtitle;
    Widget? trailingAction;

    switch (_state) {
      case LoadState.loading:
        badgeBg = isDark ? const Color(0xFF382E1A) : const Color(0xFFFFF7E6);
        badgeFg = const Color(0xFFD97706);
        badgeIcon = Icons.hourglass_top_rounded;
        title = 'Memuat Model AI';
        subtitle = 'Menyiapkan engine ONNX Runtime & bobot model...';
        break;
      case LoadState.ready:
        badgeBg = isDark ? const Color(0xFF173322) : const Color(0xFFE8F5E9);
        badgeFg = const Color(0xFF1E824C);
        badgeIcon = Icons.verified_rounded;
        title = 'AI Siap Digunakan';
        subtitle = 'Mendukung ${FlowerClassifier.numClasses} spesies bunga';
        break;
      case LoadState.parsing:
        badgeBg = isDark ? const Color(0xFF332014) : const Color(0xFFFFF3E6);
        badgeFg = const Color(0xFFE88E38);
        badgeIcon = Icons.transform_rounded;
        title = 'Memproses Gambar';
        subtitle = _parsePhase.isNotEmpty ? _parsePhase : 'Membaca dan menyiapkan gambar...';
        break;
      case LoadState.inferencing:
        badgeBg = isDark ? const Color(0xFF14293D) : const Color(0xFFE6F4FE);
        badgeFg = const Color(0xFF0284C7);
        badgeIcon = Icons.auto_awesome_rounded;
        title = 'Menganalisis Bunga';
        subtitle = _parsePhase.isNotEmpty ? _parsePhase : 'Klasifikasi dengan ONNX Runtime...';
        break;
      case LoadState.error:
        badgeBg = isDark ? const Color(0xFF3D1919) : const Color(0xFFFDE8E8);
        badgeFg = const Color(0xFFDC2626);
        badgeIcon = Icons.error_outline_rounded;
        title = 'Terjadi Kesalahan';
        subtitle = _error ?? 'Model gagal dimuat';
        trailingAction = IconButton(
          icon: const Icon(Icons.refresh_rounded),
          tooltip: 'Coba Lagi',
          onPressed: _loadModel,
          color: badgeFg,
        );
        break;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: badgeBg,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: badgeFg.withValues(alpha: 0.25), width: 1.2),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: badgeFg.withValues(alpha: 0.15),
              shape: BoxShape.circle,
            ),
            child: Icon(badgeIcon, color: badgeFg, size: 20),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: badgeFg,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w400,
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.75),
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          ?trailingAction,
        ],
      ),
    );
  }

  Widget _buildActionButtons(BuildContext context, {required bool canPick}) {
    final theme = Theme.of(context);
    final primaryColor = theme.colorScheme.primary;

    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: canPick ? () => _pick(ImageSource.camera, 'Kamera') : null,
            icon: const Icon(Icons.camera_alt_rounded, size: 20),
            label: const Text('Ambil Foto', style: TextStyle(fontWeight: FontWeight.w600)),
            style: ElevatedButton.styleFrom(
              backgroundColor: primaryColor,
              foregroundColor: Colors.white,
              elevation: 0,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              disabledBackgroundColor: primaryColor.withValues(alpha: 0.3),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: OutlinedButton.icon(
            onPressed: canPick ? () => _pick(ImageSource.gallery, 'Galeri') : null,
            icon: const Icon(Icons.photo_library_rounded, size: 20),
            label: const Text('Pilih Galeri', style: TextStyle(fontWeight: FontWeight.w600)),
            style: OutlinedButton.styleFrom(
              foregroundColor: theme.colorScheme.onSurface,
              padding: const EdgeInsets.symmetric(vertical: 14),
              side: BorderSide(color: theme.colorScheme.outline.withValues(alpha: 0.3)),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildPhotoCard(BuildContext context, {required bool busy}) {
    return Stack(
      children: [
        Container(
          width: double.infinity,
          height: 280,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(22),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.08),
                blurRadius: 16,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(22),
            child: Stack(
              fit: StackFit.expand,
              children: [
                Image.memory(
                  _photo!,
                  fit: BoxFit.cover,
                  gaplessPlayback: true,
                ),
                if (busy)
                  Container(
                    color: Colors.black.withValues(alpha: 0.55),
                    padding: const EdgeInsets.symmetric(horizontal: 24),
                    child: Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const SizedBox(
                            width: 38,
                            height: 38,
                            child: CircularProgressIndicator(
                              valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                              strokeWidth: 3,
                            ),
                          ),
                          const SizedBox(height: 16),
                          Text(
                            _state == LoadState.parsing ? 'Memproses Gambar...' : 'Klasifikasi AI...',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 15,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.2,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _parsePhase,
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
        if (!busy)
          Positioned(
            top: 12,
            right: 12,
            child: Material(
              color: Colors.black54,
              shape: const CircleBorder(),
              clipBehavior: Clip.antiAlias,
              child: IconButton(
                icon: const Icon(Icons.close_rounded, color: Colors.white, size: 20),
                tooltip: 'Hapus Foto',
                onPressed: _clearImage,
              ),
            ),
          ),
        if (!busy && _source.isNotEmpty)
          Positioned(
            bottom: 12,
            left: 12,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.6),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    _source == 'Kamera' ? Icons.camera_alt_outlined : Icons.photo_library_outlined,
                    color: Colors.white,
                    size: 14,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    _source,
                    style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w500),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildParsingPlaceholder(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: double.infinity,
      height: 220,
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: theme.colorScheme.outline.withValues(alpha: 0.2),
          width: 1.2,
        ),
      ),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(
              width: 36,
              height: 36,
              child: CircularProgressIndicator(
                strokeWidth: 3,
                valueColor: AlwaysStoppedAnimation<Color>(Color(0xFFE88E38)),
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              'Mempersiapkan Foto...',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 4),
            Text(
              _parsePhase,
              style: TextStyle(
                fontSize: 12,
                color: theme.colorScheme.onSurface.withValues(alpha: 0.65),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAnalyzingState(BuildContext context) {
    final theme = Theme.of(context);
    final isParsing = _state == LoadState.parsing;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: Container(
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.35),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(
            color: theme.colorScheme.outline.withValues(alpha: 0.15),
          ),
        ),
        child: Column(
          children: [
            Row(
              children: [
                SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(
                    strokeWidth: 2.5,
                    valueColor: AlwaysStoppedAnimation<Color>(
                      isParsing ? const Color(0xFFE88E38) : theme.colorScheme.primary,
                    ),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        isParsing ? 'Tahap 1: Decoding & Resize Gambar' : 'Tahap 2: Klasifikasi Deep Learning',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: isParsing ? const Color(0xFFE88E38) : theme.colorScheme.primary,
                        ),
                      ),
                      const SizedBox(height: 3),
                      Text(
                        _parsePhase,
                        style: TextStyle(
                          fontSize: 12,
                          color: theme.colorScheme.onSurface.withValues(alpha: 0.7),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTopResultCard(BuildContext context, Prediction top) {
    final theme = Theme.of(context);
    final pct = formatConfidence(top.score);
    final isConfident = top.score >= 0.50;

    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: theme.brightness == Brightness.dark
              ? [const Color(0xFF1B3827), const Color(0xFF13281C)]
              : [const Color(0xFFE8F6EE), const Color(0xFFF2FBF6)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: theme.brightness == Brightness.dark
              ? const Color(0xFF2E6342)
              : const Color(0xFFBCE3CB),
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: (theme.brightness == Brightness.dark ? Colors.black : Colors.green.shade100)
                .withValues(alpha: 0.35),
            blurRadius: 18,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: isConfident
                      ? const Color(0xFF1E824C).withValues(alpha: 0.15)
                      : Colors.orange.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isConfident ? Icons.verified_rounded : Icons.help_outline_rounded,
                      size: 14,
                      color: isConfident ? const Color(0xFF1E824C) : Colors.orange.shade800,
                    ),
                    const SizedBox(width: 5),
                    Text(
                      'PREDIKSI UTAMA',
                      style: TextStyle(
                        color: isConfident ? const Color(0xFF1E824C) : Colors.orange.shade800,
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 0.6,
                      ),
                    ),
                  ],
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                decoration: BoxDecoration(
                  color: isConfident ? const Color(0xFF1E824C) : Colors.orange.shade800,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '$pct%',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w800,
                    fontSize: 13,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Text(
            top.info.name,
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.5,
            ),
          ),
          const SizedBox(height: 6),
          // Class name bawaan model (mapping indeks dari class_names.json).
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            decoration: BoxDecoration(
              color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.7),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              top.info.folder,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                fontFamily: 'monospace',
                color: theme.colorScheme.onSurface.withValues(alpha: 0.75),
              ),
            ),
          ),
          if (top.info.latin.isNotEmpty) ...[
            const SizedBox(height: 3),
            Text(
              top.info.latin,
              style: TextStyle(
                fontStyle: FontStyle.italic,
                fontSize: 14,
                color: theme.colorScheme.onSurface.withValues(alpha: 0.7),
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
          if (top.info.english.isNotEmpty) ...[
            const SizedBox(height: 4),
            Row(
              children: [
                Icon(
                  Icons.translate_rounded,
                  size: 13,
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
                ),
                const SizedBox(width: 5),
                Text(
                  top.info.english,
                  style: TextStyle(
                    fontSize: 13,
                    color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
                  ),
                ),
              ],
            ),
          ],
          const SizedBox(height: 16),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: top.score,
              minHeight: 8,
              backgroundColor: (theme.brightness == Brightness.dark
                      ? Colors.white12
                      : Colors.black.withValues(alpha: 0.08)),
              valueColor: AlwaysStoppedAnimation<Color>(
                isConfident ? const Color(0xFF1E824C) : Colors.orange.shade700,
              ),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Icon(
                Icons.speed_rounded,
                size: 14,
                color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
              ),
              const SizedBox(width: 6),
              Text(
                'Inferensi: ${_inferMs}ms',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
                ),
              ),
              const Spacer(),
              Text(
                'Kategori #${top.index + 1}',
                style: TextStyle(
                  fontSize: 12,
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.5),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAlternativeHeader(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
      child: Row(
        children: [
          Icon(
            Icons.format_list_bulleted_rounded,
            size: 16,
            color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
          ),
          const SizedBox(width: 8),
          Text(
            'Kandidat Alternatif',
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: theme.colorScheme.onSurface.withValues(alpha: 0.8),
              letterSpacing: 0.2,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPredictionTile(BuildContext context, Prediction p) {
    final theme = Theme.of(context);
    final pct = formatConfidence(p.score);
    final subtitle = [
      if (p.info.english.isNotEmpty) p.info.english,
      if (p.info.latin.isNotEmpty) p.info.latin,
    ].join(' • ');

    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        p.info.name,
                        style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        p.info.folder,
                        style: TextStyle(
                          fontSize: 11,
                          fontFamily: 'monospace',
                          fontWeight: FontWeight.w600,
                          color: theme.colorScheme.onSurface.withValues(alpha: 0.55),
                        ),
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    '$pct%',
                    style: TextStyle(
                      fontWeight: FontWeight.w700,
                      fontSize: 12,
                      color: theme.colorScheme.primary,
                    ),
                  ),
                ),
              ],
            ),
            if (subtitle.isNotEmpty) ...[
              const SizedBox(height: 3),
              Text(
                subtitle,
                style: TextStyle(
                  fontSize: 12,
                  fontStyle: FontStyle.italic,
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
                ),
              ),
            ],
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: p.score,
                minHeight: 5,
                backgroundColor: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
                valueColor: AlwaysStoppedAnimation<Color>(
                  theme.colorScheme.primary.withValues(alpha: 0.75),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyState(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 36),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(22),
              decoration: BoxDecoration(
                color: theme.colorScheme.primary.withValues(alpha: 0.08),
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.center_focus_strong_rounded,
                size: 48,
                color: theme.colorScheme.primary.withValues(alpha: 0.8),
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              'Belum Ada Foto Bunga',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 6),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32),
              child: Text(
                'Ambil foto dengan kamera atau unggah dari galeri untuk mulai mengenali spesies bunga secara instan.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 13,
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
                  height: 1.4,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SpeciesCatalogSheet extends StatefulWidget {
  const _SpeciesCatalogSheet({required this.classifier});

  final FlowerClassifier classifier;

  @override
  State<_SpeciesCatalogSheet> createState() => _SpeciesCatalogSheetState();
}

class _SpeciesCatalogSheetState extends State<_SpeciesCatalogSheet> {
  final _searchController = TextEditingController();
  String _query = '';

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final all = List.generate(
      FlowerClassifier.numClasses,
      (i) => widget.classifier.infoFor(i),
    );

    final filtered = _query.isEmpty
        ? all
        : all.where((f) {
            final q = _query.toLowerCase();
            return f.name.toLowerCase().contains(q) ||
                f.english.toLowerCase().contains(q) ||
                f.latin.toLowerCase().contains(q);
          }).toList();

    return DraggableScrollableSheet(
      initialChildSize: 0.85,
      maxChildSize: 0.95,
      minChildSize: 0.5,
      builder: (ctx, scrollController) {
        return Container(
          decoration: BoxDecoration(
            color: theme.scaffoldBackgroundColor,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
          ),
          child: Column(
            children: [
              const SizedBox(height: 12),
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: theme.colorScheme.onSurface.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Katalog Spesies Bunga',
                            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
                          ),
                          Text(
                            '${all.length} spesies terdaftar dalam model',
                            style: TextStyle(
                              fontSize: 12,
                              color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
                            ),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close_rounded),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                child: TextField(
                  controller: _searchController,
                  decoration: InputDecoration(
                    hintText: 'Cari nama Indonesia, Inggris, atau Latin...',
                    prefixIcon: const Icon(Icons.search_rounded, size: 20),
                    suffixIcon: _query.isNotEmpty
                        ? IconButton(
                            icon: const Icon(Icons.clear_rounded, size: 18),
                            onPressed: () {
                              _searchController.clear();
                              setState(() => _query = '');
                            },
                          )
                        : null,
                    filled: true,
                    fillColor: theme.cardTheme.color,
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(14),
                      borderSide: BorderSide.none,
                    ),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  ),
                  onChanged: (val) => setState(() => _query = val),
                ),
              ),
              const SizedBox(height: 8),
              Expanded(
                child: filtered.isEmpty
                    ? Center(
                        child: Text(
                          'Tidak ada spesies yang cocok dengan "$_query"',
                          style: TextStyle(color: theme.colorScheme.onSurface.withValues(alpha: 0.5)),
                        ),
                      )
                    : ListView.separated(
                        controller: scrollController,
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                        itemCount: filtered.length,
                        separatorBuilder: (context, index) => const SizedBox(height: 8),
                        itemBuilder: (context, idx) {
                          final flower = filtered[idx];
                          final sub = [
                            if (flower.latin.isNotEmpty) flower.latin,
                            if (flower.english.isNotEmpty) flower.english,
                          ].join(' • ');

                          return Card(
                            margin: EdgeInsets.zero,
                            child: ListTile(
                              leading: CircleAvatar(
                                backgroundColor: theme.colorScheme.primary.withValues(alpha: 0.12),
                                foregroundColor: theme.colorScheme.primary,
                                child: const Icon(Icons.local_florist_rounded, size: 20),
                              ),
                              title: Text(
                                flower.name,
                                style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15),
                              ),
                              subtitle: sub.isNotEmpty
                                  ? Text(
                                      sub,
                                      style: TextStyle(
                                        fontSize: 12,
                                        fontStyle: FontStyle.italic,
                                        color: theme.colorScheme.onSurface.withValues(alpha: 0.65),
                                      ),
                                    )
                                  : null,
                              trailing: Text(
                                '#${idx + 1}',
                                style: TextStyle(
                                  fontSize: 11,
                                  color: theme.colorScheme.onSurface.withValues(alpha: 0.4),
                                ),
                              ),
                            ),
                          );
                        },
                      ),
              ),
            ],
          ),
        );
      },
    );
  }
}
