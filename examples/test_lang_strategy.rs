use obs_live_translator::{
    language_detection::MultiDetector,
    types::{LanguageDetectionStrategy, TranslatorConfig},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Test Hybrid Strategy (Default)
    println!("Testing Hybrid Strategy...");
    let mut config = TranslatorConfig::default();
    config.language_detection_strategy = LanguageDetectionStrategy::Hybrid;

    let detector = MultiDetector::new(&config).await?;

    // Case 1: ASR hint provided -> Should use hint
    let result = detector.detect("Bonjour", Some("fr")).await?;
    assert_eq!(result, "fr", "Hybrid should respect ASR hint");
    println!("✅ Hybrid with hint passed");

    // Case 2: No hint -> Should use text detection
    let result = detector
        .detect("Bonjour tout le monde, comment ça va aujourd'hui?", None)
        .await?;
    assert_eq!(result, "fr", "Hybrid without hint should detect text");
    println!("✅ Hybrid without hint passed");

    // Test TextOnly Strategy
    println!("\nTesting TextOnly Strategy...");
    config.language_detection_strategy = LanguageDetectionStrategy::TextOnly;
    let detector = MultiDetector::new(&config).await?;

    // Case 3: Hint provided but strategy is TextOnly -> Should ignore hint
    // We pass a WRONG hint ("en") for French text to verify it ignores it
    let result = detector
        .detect(
            "Bonjour tout le monde, comment ça va aujourd'hui?",
            Some("en"),
        )
        .await?;
    assert_eq!(result, "fr", "TextOnly should ignore ASR hint");
    println!("✅ TextOnly passed");

    // Test AsrOnly Strategy
    println!("\nTesting AsrOnly Strategy...");
    config.language_detection_strategy = LanguageDetectionStrategy::AsrOnly;
    let detector = MultiDetector::new(&config).await?;

    // Case 4: Hint provided -> Use it
    let result = detector.detect("Bonjour", Some("fr")).await?;
    assert_eq!(result, "fr", "AsrOnly should use hint");
    println!("✅ AsrOnly with hint passed");

    // Case 5: No hint -> Fallback (implementation details: currently falls back to text detection in code)
    let result = detector
        .detect("Bonjour tout le monde, comment ça va aujourd'hui?", None)
        .await?;
    assert_eq!(result, "fr", "AsrOnly fallback passed");
    println!("✅ AsrOnly fallback passed");

    Ok(())
}
