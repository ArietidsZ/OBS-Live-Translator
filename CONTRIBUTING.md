# Contributing to OBS Live Translator

Thank you for your interest in contributing to OBS Live Translator! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Rust 1.70 or higher
- Cargo package manager
- Git for version control
- (Optional) NVIDIA GPU with CUDA 11.8+ for GPU acceleration

### Development Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/obs-live-translator.git
   cd obs-live-translator
   ```

2. **Install Rust toolchain**
   ```bash
   rustup update stable
   rustup component add clippy rustfmt
   ```

3. **Install development tools**
   ```bash
   cargo install cargo-watch cargo-audit cargo-tarpaulin
   ```

4. **Build the project**
   ```bash
   cargo build
   ```

5. **Run tests**
   ```bash
   cargo test
   ```

## Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `perf/description` - Performance improvements
- `test/description` - Test additions or modifications

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `test` - Test additions or fixes
- `chore` - Build process or auxiliary tool changes

**Example:**
```
feat(audio): add SIMD-optimized VAD implementation

Implement WebRTC VAD with AVX-512 optimizations for low-latency
voice activity detection on high-performance profiles.

Closes #123
```

## Coding Standards

### Rust Style Guide

We follow the official [Rust Style Guide](https://doc.rust-lang.org/nightly/style-guide/) with some project-specific additions:

#### Formatting

- Use `rustfmt` for code formatting
- Run `cargo fmt` before committing
- Maximum line length: 100 characters
- Use 4 spaces for indentation

#### Naming Conventions

- Types: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Module files: `snake_case.rs`

#### Documentation

- All public APIs must have documentation comments (`///`)
- Module-level documentation should use `//!`
- Include usage examples for complex functions
- Document error conditions and panics

**Example:**
```rust
/// Process audio samples and return transcription result
///
/// # Arguments
///
/// * `audio_samples` - Slice of f32 audio samples at 16kHz
///
/// # Returns
///
/// `TranslationResult` containing transcription and translation
///
/// # Errors
///
/// Returns error if VAD processing fails or model inference errors occur
///
/// # Examples
///
/// ```
/// let translator = Translator::new("model.onnx", false)?;
/// let result = translator.process_audio(&samples).await?;
/// println!("Transcription: {}", result.original_text);
/// ```
pub async fn process_audio(&mut self, audio_samples: &[f32]) -> Result<TranslationResult>
```

#### Error Handling

- Use `Result<T, E>` for fallible operations
- Use `anyhow::Result` for application-level errors
- Create custom error types for library APIs
- Always provide context with `anyhow::Context`

#### Performance Guidelines

- Profile before optimizing
- Benchmark performance-critical code
- Use `#[inline]` judiciously for hot paths
- Prefer zero-copy operations where possible
- Use proper memory allocation strategies (arena, pool, etc.)

#### Memory Safety

- Avoid `unsafe` unless necessary
- Document all unsafe code with safety invariants
- Run memory leak detection on audio pipeline changes
- Use RAII patterns for resource management

## Testing

### Test Categories

1. **Unit Tests** - Test individual functions and modules
2. **Integration Tests** - Test component interactions
3. **Performance Tests** - Verify latency and throughput requirements
4. **Memory Tests** - Detect memory leaks and validate usage

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test suite
cargo test --test integration_tests

# Run with output
cargo test -- --nocapture

# Run benchmarks
cargo bench
```

### Test Coverage

- Maintain >80% code coverage
- All public APIs must have tests
- Performance-critical paths require benchmarks

```bash
# Generate coverage report
cargo tarpaulin --out Html
```

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_processing() {
        let samples = vec![0.0f32; 1024];
        let result = process_samples(&samples);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_translation() {
        let translator = Translator::new("model.onnx", false).unwrap();
        let result = translator.process_audio(&samples).await;
        assert!(result.is_ok());
    }
}
```

## Documentation

### Types of Documentation

1. **API Documentation** - Rustdoc comments for all public items
2. **User Guides** - Markdown files in `docs/`
3. **Examples** - Working code in `examples/`
4. **Architecture** - High-level design documentation

### Documentation Requirements

- All public functions, structs, and modules must be documented
- Include at least one usage example for complex APIs
- Update relevant markdown files when adding features
- Keep README.md synchronized with capabilities

### Building Documentation

```bash
# Generate API documentation
cargo doc --no-deps --open

# Build with all features
cargo doc --all-features --no-deps
```

## Pull Request Process

### Before Submitting

1. **Update your branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout your-branch
   git rebase main
   ```

2. **Run quality checks**
   ```bash
   # Format code
   cargo fmt

   # Lint code
   cargo clippy -- -D warnings

   # Run tests
   cargo test

   # Security audit
   cargo audit
   ```

3. **Update documentation**
   - Update API docs for changed functions
   - Update relevant markdown files
   - Add examples if introducing new features

4. **Performance verification**
   ```bash
   # Run benchmarks
   cargo bench

   # Check for regressions
   cargo run --bin benchmark
   ```

### PR Title Format

Use the same format as commit messages:
```
<type>(<scope>): <description>
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Benchmarks run (include results)
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No performance regression

## Performance Impact
Include benchmark results for performance-critical changes

## Breaking Changes
List any breaking changes and migration path
```

### Review Process

1. All PRs require at least one approval
2. CI checks must pass
3. No unresolved comments
4. Up-to-date with main branch

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
Clear and concise description

**To Reproduce**
Steps to reproduce the behavior

**Expected behavior**
What you expected to happen

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Rust version: [e.g., 1.70]
- Hardware: [CPU, RAM, GPU]
- Profile: [Low/Medium/High]

**Logs**
Include relevant log output

**Screenshots**
If applicable
```

### Feature Requests

```markdown
**Is your feature request related to a problem?**
Clear description of the problem

**Describe the solution**
Proposed solution

**Alternatives considered**
Alternative solutions you've considered

**Additional context**
Any other context or screenshots
```

## Performance Guidelines

### Benchmarking Requirements

All performance-critical changes must include benchmarks:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_audio_processing(c: &mut Criterion) {
    let samples = vec![0.0f32; 16000];
    c.bench_function("process_audio", |b| {
        b.iter(|| process_audio(black_box(&samples)))
    });
}

criterion_group!(benches, benchmark_audio_processing);
criterion_main!(benches);
```

### Performance Targets

- **Low Profile**: <500ms latency, <1GB memory
- **Medium Profile**: <400ms latency, <4GB total memory
- **High Profile**: <250ms latency, <10GB total memory

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

Feel free to:
- Open a discussion on GitHub
- Join our community forum
- Ask in pull request comments

Thank you for contributing to OBS Live Translator!
