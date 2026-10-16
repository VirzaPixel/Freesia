import 'package:flutter_test/flutter_test.dart';

import 'package:freesia_detector/classifier.dart';
import 'package:freesia_detector/main.dart';

void main() {
  test('softmax ter-normalisasi dan monoton', () {
    final probs = FlowerClassifier.softmaxForTest([1.0, 2.0, 3.0]);
    expect(probs.fold<double>(0, (a, b) => a + b), closeTo(1.0, 1e-9));
    expect(probs[2], greaterThan(probs[1]));
    expect(probs[1], greaterThan(probs[0]));
  });

  test('parseClassNames mengurutkan indeks numerik dari mapping JSON', () {
    const raw = '{"1":"02_Jasmine","0":"01_Rose","10":"10_X","2":"03_Tulip"}';
    final labels = FlowerClassifier.parseClassNames(raw);
    expect(labels, ['01_Rose', '02_Jasmine', '03_Tulip', '10_X']);
  });

  test('parseInfo membaca nama Indonesia dan Latin', () {
    const raw = '[{"folder":"01_Rose","name":"Mawar",'
        '"en":"Rose","latin":"Rosa","fallback":"Rose"}]';
    final info = FlowerClassifier.parseInfo(raw);
    expect(info['01_Rose']!.name, 'Mawar');
    expect(info['01_Rose']!.latin, 'Rosa');
  });

  test('formatConfidence formats exact percentage with 2 decimal places', () {
    expect(formatConfidence(0.9779), '97.79');
    expect(formatConfidence(0.0062), '0.62');
    expect(formatConfidence(0.0013), '0.13');
    expect(formatConfidence(0.0005), '0.05');
  });

  testWidgets('FreesiaApp renders header and initial action buttons', (tester) async {
    await tester.pumpWidget(const FreesiaApp());
    await tester.pump();

    expect(find.text('Freesia'), findsOneWidget);
    expect(find.text('Botanical Vision AI'), findsOneWidget);
    expect(find.text('Ambil Foto'), findsOneWidget);
    expect(find.text('Pilih Galeri'), findsOneWidget);
  });
}
