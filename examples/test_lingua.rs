#[cfg(feature = "lingua-fallback")]
use obs_live_translator::language_detection::lingua_detector::LinguaDetector;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    #[cfg(not(feature = "lingua-fallback"))]
    {
        println!("Lingua feature not enabled. Run with --features lingua-fallback");
    // return Ok(());
    }

    #[cfg(feature = "lingua-fallback")]
    {
        println!("Initializing Lingua Detector (this may take a moment to load models)...");
        let start_init = Instant::now();
        let detector = LinguaDetector::new()?;
        println!("Initialization took: {:?}", start_init.elapsed());

        let test_cases = vec![
            ("Hello world, this is a test.", "en"),
            ("Bonjour tout le monde, comment ça va?", "fr"),
            ("Hola mundo, esto es una prueba.", "es"),
            ("你好世界，这是一个测试。", "zh"),
            ("Привет мир, это тест.", "ru"),
            ("مرحبا بالعالم", "ar"),
        ];

        println!("\nRunning accuracy tests:");
        for (text, expected) in &test_cases {
            let start = Instant::now();
            let result = detector.detect(text)?;
            let duration = start.elapsed();

            println!("Text: '{}'", text);
            println!(
                "  Detected: {} (Confidence: {:.4})",
                result.language, result.confidence
            );
            println!("  Expected: {}", expected);
            println!("  Time: {:?}", duration);

            let is_match = result.language.starts_with(expected);
            if !is_match {
                println!("  WARNING: Mismatch!");
            }
        }

        println!("\nRunning benchmark (100 iterations)...");
        let text =
            "This is a sample text to measure the performance of the language detection model.";
        let start_bench = Instant::now();
        for _ in 0..100 {
            let _ = detector.detect(text)?;
        }
        let total_duration = start_bench.elapsed();
        println!("Total time: {:?}", total_duration);
        println!("Average time per detection: {:?}", total_duration / 100);
    }

    Ok(())
}
