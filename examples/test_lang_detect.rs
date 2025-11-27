use obs_live_translator::language_detection::whatlang_detector::WhatlangDetector;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("Initializing Whatlang Detector...");
    let start_init = Instant::now();
    let detector = WhatlangDetector::new()?;
    println!("Initialization took: {:?}", start_init.elapsed());

    let test_cases = vec![
        ("Hello world, this is a test.", "en"),
        ("Bonjour tout le monde, comment ça va?", "fr"),
        ("Hola mundo, esto es una prueba.", "es"),
        ("你好世界，这是一个测试。", "zh"), // whatlang usually returns "cmn" for Chinese
        ("Привет мир, это тест.", "ru"),
        ("مرحبا بالعالم", "ar"),
    ];

    println!("\nRunning accuracy tests:");
    for (text, expected) in &test_cases {
        let start = Instant::now();
        let result = detector.detect(text)?;
        let duration = start.elapsed();

        println!("Text: '{text}'");
        println!(
            "  Detected: {} (Confidence: {:.4})",
            result.language, result.confidence
        );
        println!("  Expected: {expected}");
        println!("  Time: {duration:?}");

        // Basic validation
        let is_match = result.language.starts_with(expected)
            || (expected == &"zh" && (result.language == "cmn" || result.language == "zh"));

        if !is_match {
            println!("  WARNING: Mismatch!");
        }
    }

    println!("\nRunning benchmark (10000 iterations)...");
    let text = "This is a sample text to measure the performance of the language detection model.";
    let start_bench = Instant::now();
    for _ in 0..10000 {
        let _ = detector.detect(text)?;
    }
    let total_duration = start_bench.elapsed();
    println!("Total time: {total_duration:?}");
    println!("Average time per detection: {:?}", total_duration / 10000);

    Ok(())
}
