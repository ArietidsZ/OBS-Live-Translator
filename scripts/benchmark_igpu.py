#!/usr/bin/env python3
"""
Integrated GPU Benchmark Tool for OBS Live Translator
Measures actual inference latency on integrated GPUs
"""

import os
import sys
import time
import json
import numpy as np
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import psutil

# Try to import ML libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available")

class IGPUBenchmark:
    """Benchmark integrated GPU performance for translation models"""

    def __init__(self):
        self.results = {
            "system": self.get_system_info(),
            "gpu": self.detect_gpu(),
            "memory": self.get_memory_info(),
            "benchmarks": {}
        }

    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1)
        }

    def detect_gpu(self) -> Dict:
        """Detect integrated GPU type and capabilities"""
        gpu_info = {
            "vendor": "Unknown",
            "name": "Unknown",
            "tflops": 0,
            "memory_mb": 0,
            "type": "unknown"
        }

        system = platform.system()

        if system == "Darwin":
            # macOS - detect Apple Silicon
            try:
                chip_info = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True
                ).strip()

                if "Apple" in chip_info:
                    gpu_info["vendor"] = "Apple"
                    gpu_info["type"] = "integrated"

                    if "M1" in chip_info:
                        if "Max" in chip_info:
                            gpu_info["name"] = "Apple M1 Max GPU"
                            gpu_info["tflops"] = 10.4
                        elif "Pro" in chip_info:
                            gpu_info["name"] = "Apple M1 Pro GPU"
                            gpu_info["tflops"] = 5.2
                        else:
                            gpu_info["name"] = "Apple M1 GPU"
                            gpu_info["tflops"] = 2.6
                    elif "M2" in chip_info:
                        if "Max" in chip_info:
                            gpu_info["name"] = "Apple M2 Max GPU"
                            gpu_info["tflops"] = 13.6
                        elif "Pro" in chip_info:
                            gpu_info["name"] = "Apple M2 Pro GPU"
                            gpu_info["tflops"] = 6.8
                        else:
                            gpu_info["name"] = "Apple M2 GPU"
                            gpu_info["tflops"] = 3.6
                    elif "M3" in chip_info:
                        if "Max" in chip_info:
                            gpu_info["name"] = "Apple M3 Max GPU"
                            gpu_info["tflops"] = 14.2
                        elif "Pro" in chip_info:
                            gpu_info["name"] = "Apple M3 Pro GPU"
                            gpu_info["tflops"] = 7.0
                        else:
                            gpu_info["name"] = "Apple M3 GPU"
                            gpu_info["tflops"] = 4.5

                    # Estimate available memory (40% of system RAM)
                    total_ram = psutil.virtual_memory().total
                    gpu_info["memory_mb"] = int(total_ram * 0.4 / (1024**2))

            except Exception as e:
                print(f"Error detecting Apple Silicon: {e}")

        elif system == "Linux" or system == "Windows":
            # Try to detect Intel/AMD integrated GPUs
            try:
                if system == "Linux":
                    lspci_output = subprocess.check_output(
                        ["lspci"], text=True
                    )

                    for line in lspci_output.split('\n'):
                        if 'VGA' in line or 'Display' in line or '3D' in line:
                            line_lower = line.lower()

                            if 'intel' in line_lower:
                                gpu_info["vendor"] = "Intel"
                                gpu_info["type"] = "integrated"

                                if 'arc' in line_lower:
                                    gpu_info["name"] = "Intel Arc Graphics"
                                    gpu_info["tflops"] = 4.6
                                    gpu_info["memory_mb"] = 4096
                                elif 'iris xe' in line_lower:
                                    if '96eu' in line_lower:
                                        gpu_info["name"] = "Intel Iris Xe 96EU"
                                        gpu_info["tflops"] = 2.4
                                    else:
                                        gpu_info["name"] = "Intel Iris Xe 80EU"
                                        gpu_info["tflops"] = 2.0
                                    gpu_info["memory_mb"] = 3072
                                elif 'uhd' in line_lower:
                                    if '770' in line_lower or '750' in line_lower:
                                        gpu_info["name"] = "Intel UHD 770"
                                        gpu_info["tflops"] = 1.5
                                    else:
                                        gpu_info["name"] = "Intel UHD Graphics"
                                        gpu_info["tflops"] = 0.8
                                    gpu_info["memory_mb"] = 2048

                            elif 'amd' in line_lower and 'radeon' in line_lower:
                                gpu_info["vendor"] = "AMD"
                                gpu_info["type"] = "integrated"

                                if '780m' in line_lower:
                                    gpu_info["name"] = "AMD Radeon 780M"
                                    gpu_info["tflops"] = 8.9
                                    gpu_info["memory_mb"] = 4096
                                elif '760m' in line_lower:
                                    gpu_info["name"] = "AMD Radeon 760M"
                                    gpu_info["tflops"] = 4.3
                                    gpu_info["memory_mb"] = 3072
                                elif '680m' in line_lower:
                                    gpu_info["name"] = "AMD Radeon 680M"
                                    gpu_info["tflops"] = 3.4
                                    gpu_info["memory_mb"] = 3072
                                elif '660m' in line_lower:
                                    gpu_info["name"] = "AMD Radeon 660M"
                                    gpu_info["tflops"] = 1.8
                                    gpu_info["memory_mb"] = 2048
                                elif 'vega' in line_lower:
                                    gpu_info["name"] = "AMD Radeon Vega"
                                    gpu_info["tflops"] = 1.1
                                    gpu_info["memory_mb"] = 2048

                            elif 'nvidia' in line_lower:
                                # Discrete GPU detected
                                gpu_info["vendor"] = "NVIDIA"
                                gpu_info["type"] = "discrete"
                                # Would need nvidia-smi for details

                except Exception as e:
                    print(f"Error detecting GPU on {system}: {e}")

        # Detect available execution providers
        gpu_info["execution_providers"] = self.detect_execution_providers()

        return gpu_info

    def detect_execution_providers(self) -> List[str]:
        """Detect available ONNX Runtime execution providers"""
        providers = []

        if ONNX_AVAILABLE:
            available = ort.get_available_providers()

            # Map providers to friendly names
            provider_map = {
                'TensorrtExecutionProvider': 'TensorRT',
                'CUDAExecutionProvider': 'CUDA',
                'ROCMExecutionProvider': 'ROCm',
                'MIGraphXExecutionProvider': 'MIGraphX',
                'OpenVINOExecutionProvider': 'OpenVINO',
                'DmlExecutionProvider': 'DirectML',
                'CoreMLExecutionProvider': 'CoreML',
                'CPUExecutionProvider': 'CPU'
            }

            for provider in available:
                if provider in provider_map:
                    providers.append(provider_map[provider])

        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                providers.append("PyTorch CUDA")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                providers.append("PyTorch MPS")

        return providers

    def get_memory_info(self) -> Dict:
        """Get memory information"""
        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / (1024**3), 1),
            "available_gb": round(vm.available / (1024**3), 1),
            "used_percent": vm.percent
        }

    def benchmark_matrix_multiply(self, size: int = 1024) -> float:
        """Benchmark matrix multiplication performance"""
        if not TORCH_AVAILABLE:
            return 0

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"

        # Create random matrices
        a = torch.randn(size, size, dtype=torch.float32, device=device)
        b = torch.randn(size, size, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(5):
            _ = torch.matmul(a, b)

        if device != "cpu":
            torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()

        # Benchmark
        iterations = 100
        start = time.perf_counter()

        for _ in range(iterations):
            _ = torch.matmul(a, b)

        if device != "cpu":
            torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()

        elapsed = time.perf_counter() - start

        # Calculate GFLOPS
        ops = 2 * size * size * size * iterations  # Matrix multiply ops
        gflops = ops / (elapsed * 1e9)

        return gflops

    def benchmark_whisper_simulation(self) -> Dict:
        """Simulate Whisper model inference"""
        results = {}

        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"

        # Simulate different Whisper model sizes
        models = {
            "tiny": {"encoder_dim": 384, "decoder_dim": 384, "layers": 4},
            "base": {"encoder_dim": 512, "decoder_dim": 512, "layers": 6},
            "small": {"encoder_dim": 768, "decoder_dim": 768, "layers": 12}
        }

        for model_name, config in models.items():
            # Simulate encoder (audio processing)
            batch_size = 1
            seq_len = 1500  # ~30 seconds at 50Hz

            encoder_input = torch.randn(
                batch_size, seq_len, config["encoder_dim"],
                dtype=torch.float32, device=device
            )

            # Warmup
            for _ in range(3):
                _ = torch.nn.functional.relu(encoder_input)

            if device != "cpu":
                torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()

            # Benchmark encoder
            start = time.perf_counter()
            for _ in range(10):
                # Simulate transformer layers
                x = encoder_input
                for _ in range(config["layers"]):
                    # Self-attention simulation
                    x = torch.matmul(x, x.transpose(-2, -1))
                    x = torch.nn.functional.softmax(x, dim=-1)
                    # FFN simulation
                    x = torch.nn.functional.relu(x)

            if device != "cpu":
                torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()

            encoder_time = (time.perf_counter() - start) / 10 * 1000  # ms

            # Simulate decoder (text generation)
            decoder_seq_len = 100  # Average output tokens
            decoder_input = torch.randn(
                batch_size, decoder_seq_len, config["decoder_dim"],
                dtype=torch.float32, device=device
            )

            start = time.perf_counter()
            for _ in range(10):
                x = decoder_input
                for _ in range(config["layers"]):
                    x = torch.matmul(x, x.transpose(-2, -1))
                    x = torch.nn.functional.softmax(x, dim=-1)
                    x = torch.nn.functional.relu(x)

            if device != "cpu":
                torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()

            decoder_time = (time.perf_counter() - start) / 10 * 1000  # ms

            results[model_name] = {
                "encoder_ms": round(encoder_time, 1),
                "decoder_ms": round(decoder_time, 1),
                "total_ms": round(encoder_time + decoder_time, 1),
                "device": device
            }

        return results

    def estimate_real_world_latency(self) -> Dict:
        """Estimate real-world translation latency based on GPU performance"""
        gpu_tflops = self.results["gpu"]["tflops"]

        estimates = {}

        # Base latency estimates (ms) for different TFLOPS ranges
        if gpu_tflops >= 8.0:  # High-end iGPU (Radeon 780M)
            estimates = {
                "whisper_tiny": {"min": 40, "max": 60},
                "whisper_base": {"min": 80, "max": 120},
                "nllb_600m": {"min": 100, "max": 150},
                "total": {"min": 180, "max": 250}
            }
        elif gpu_tflops >= 4.0:  # Mid-high iGPU (Arc Graphics, Radeon 760M)
            estimates = {
                "whisper_tiny": {"min": 60, "max": 90},
                "whisper_base": {"min": 120, "max": 180},
                "nllb_600m": {"min": 100, "max": 150},
                "total": {"min": 220, "max": 300}
            }
        elif gpu_tflops >= 2.0:  # Mid-range iGPU (Iris Xe)
            estimates = {
                "whisper_tiny": {"min": 100, "max": 150},
                "whisper_base": {"min": 200, "max": 300},
                "nllb_600m": {"min": 200, "max": 250},
                "total": {"min": 400, "max": 550}
            }
        elif gpu_tflops >= 1.0:  # Low-end iGPU
            estimates = {
                "whisper_tiny": {"min": 150, "max": 250},
                "whisper_base": {"min": 300, "max": 500},
                "nllb_600m": {"min": 300, "max": 400},
                "total": {"min": 600, "max": 800}
            }
        else:  # Very low-end or CPU only
            estimates = {
                "whisper_tiny": {"min": 300, "max": 500},
                "whisper_base": {"min": 600, "max": 1000},
                "nllb_600m": {"min": 500, "max": 700},
                "total": {"min": 1000, "max": 1500}
            }

        # Add RTF (Real-Time Factor) estimates
        for model in estimates:
            if model != "total":
                # Calculate RTF for 1 second of audio
                avg_latency = (estimates[model]["min"] + estimates[model]["max"]) / 2
                estimates[model]["rtf"] = round(avg_latency / 1000, 3)

        # Overall RTF for the full pipeline
        avg_total = (estimates["total"]["min"] + estimates["total"]["max"]) / 2
        estimates["total"]["rtf"] = round(avg_total / 1000, 3)
        estimates["total"]["realtime_capable"] = estimates["total"]["rtf"] < 1.0

        return estimates

    def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on detected hardware"""
        recommendations = []

        gpu_tflops = self.results["gpu"]["tflops"]
        gpu_vendor = self.results["gpu"]["vendor"]
        memory_gb = self.results["memory"]["total_gb"]
        providers = self.results["gpu"]["execution_providers"]

        # Model selection recommendations
        if gpu_tflops < 2.0:
            recommendations.append(
                "⚠️ Use Whisper Tiny model only - larger models will cause severe lag"
            )
            recommendations.append(
                "⚠️ Consider CPU-only mode if GPU acceleration causes instability"
            )
        elif gpu_tflops < 4.0:
            recommendations.append(
                "✓ Whisper Tiny or Base models recommended"
            )
            recommendations.append(
                "✓ Use INT8 quantization for all models"
            )
        else:
            recommendations.append(
                "✓ Can handle Whisper Base model comfortably"
            )
            recommendations.append(
                "✓ Consider Whisper Small for better accuracy if latency allows"
            )

        # Memory recommendations
        if memory_gb < 16:
            recommendations.append(
                f"⚠️ Limited RAM ({memory_gb}GB) - close other applications"
            )

        # Execution provider recommendations
        if gpu_vendor == "Intel":
            if "OpenVINO" in providers:
                recommendations.append(
                    "✓ OpenVINO detected - best performance for Intel iGPU"
                )
            else:
                recommendations.append(
                    "💡 Install OpenVINO for 20-30% better performance on Intel"
                )

        elif gpu_vendor == "AMD":
            if "ROCm" in providers:
                recommendations.append(
                    "✓ ROCm detected - optimal for AMD GPUs"
                )
            elif "DirectML" in providers:
                recommendations.append(
                    "✓ DirectML available - good Windows performance"
                )
            else:
                recommendations.append(
                    "💡 Install ROCm (Linux) or use DirectML (Windows) for AMD GPU"
                )

        elif gpu_vendor == "Apple":
            if "PyTorch MPS" in providers or "CoreML" in providers:
                recommendations.append(
                    "✓ Metal/CoreML support detected - optimal for Apple Silicon"
                )
            else:
                recommendations.append(
                    "💡 Ensure PyTorch is built with MPS support for Apple Silicon"
                )

        # General optimizations
        recommendations.extend([
            "💡 Enable VAD (Voice Activity Detection) to skip silence",
            "💡 Use shorter audio chunks (250-500ms) for lower latency",
            "💡 Disable beam search and use greedy decoding",
            "💡 Consider hybrid CPU-GPU execution for memory-limited systems"
        ])

        return recommendations

    def run_benchmarks(self):
        """Run all benchmarks"""
        print("\n" + "="*60)
        print("OBS Live Translator - Integrated GPU Benchmark")
        print("="*60)

        # Display system info
        print(f"\n📊 System Information:")
        print(f"  Platform: {self.results['system']['platform']}")
        print(f"  CPU: {self.results['system']['processor']}")
        print(f"  RAM: {self.results['system']['ram_gb']} GB")

        # Display GPU info
        print(f"\n🎮 GPU Information:")
        print(f"  Vendor: {self.results['gpu']['vendor']}")
        print(f"  Model: {self.results['gpu']['name']}")
        print(f"  Type: {self.results['gpu']['type']}")
        print(f"  Performance: {self.results['gpu']['tflops']} TFLOPS")
        print(f"  Memory: {self.results['gpu']['memory_mb']} MB")
        print(f"  Providers: {', '.join(self.results['gpu']['execution_providers'])}")

        # Run matrix multiplication benchmark
        print("\n⚡ Running performance benchmarks...")

        if TORCH_AVAILABLE:
            gflops = self.benchmark_matrix_multiply()
            self.results["benchmarks"]["matrix_multiply_gflops"] = round(gflops, 2)
            print(f"  Matrix Multiply: {gflops:.2f} GFLOPS")

            # Run Whisper simulation
            whisper_results = self.benchmark_whisper_simulation()
            self.results["benchmarks"]["whisper_simulation"] = whisper_results

            print("\n📝 Whisper Model Simulation:")
            for model, timings in whisper_results.items():
                if isinstance(timings, dict) and "total_ms" in timings:
                    print(f"  {model}: {timings['total_ms']}ms "
                          f"(encoder: {timings['encoder_ms']}ms, "
                          f"decoder: {timings['decoder_ms']}ms)")

        # Estimate real-world latency
        estimates = self.estimate_real_world_latency()
        self.results["latency_estimates"] = estimates

        print("\n⏱️  Estimated Real-World Latency:")
        print(f"  Whisper Tiny: {estimates['whisper_tiny']['min']}-"
              f"{estimates['whisper_tiny']['max']}ms "
              f"(RTF: {estimates['whisper_tiny']['rtf']})")
        print(f"  Whisper Base: {estimates['whisper_base']['min']}-"
              f"{estimates['whisper_base']['max']}ms "
              f"(RTF: {estimates['whisper_base']['rtf']})")
        print(f"  NLLB-600M: {estimates['nllb_600m']['min']}-"
              f"{estimates['nllb_600m']['max']}ms")
        print(f"  Total Pipeline: {estimates['total']['min']}-"
              f"{estimates['total']['max']}ms "
              f"(RTF: {estimates['total']['rtf']})")

        if estimates['total']['realtime_capable']:
            print("  ✅ Real-time capable (RTF < 1.0)")
        else:
            print("  ⚠️ NOT real-time capable (RTF >= 1.0)")

        # Generate recommendations
        recommendations = self.generate_optimization_recommendations()
        self.results["recommendations"] = recommendations

        print("\n💡 Optimization Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")

        # Save results
        results_file = Path("benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📁 Results saved to: {results_file}")

        # Generate config recommendation
        self.generate_config_recommendation()

    def generate_config_recommendation(self):
        """Generate recommended configuration file"""
        gpu_tflops = self.results["gpu"]["tflops"]

        if gpu_tflops >= 8.0:
            config_profile = "igpu_amd"
            print("\n🔧 Recommended configuration: config/profiles/igpu_amd.toml")
        elif gpu_tflops >= 4.0:
            config_profile = "igpu_amd"  # Also works for Intel Arc
            print("\n🔧 Recommended configuration: config/profiles/igpu_amd.toml")
        elif gpu_tflops >= 2.0:
            config_profile = "igpu_intel"
            print("\n🔧 Recommended configuration: config/profiles/igpu_intel.toml")
        else:
            config_profile = "cpu_only"
            print("\n🔧 Recommended configuration: config/profiles/cpu_only.toml")

        print(f"   Run: ./scripts/optimize.sh --auto-detect")
        print(f"   Or:  cp config/profiles/{config_profile}.toml config/production.toml")

if __name__ == "__main__":
    benchmark = IGPUBenchmark()
    benchmark.run_benchmarks()