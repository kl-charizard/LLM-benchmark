# LLM Benchmark Tool

The **LLM Benchmark Tool** is a Python-based GUI application designed to evaluate the performance of various hardware accelerators—including CPUs, CUDA-enabled GPUs, and Apple MPS devices—by benchmarking multiple precision modes. It assesses floating point precisions (FP32, FP16, simulated FP8) and integer precisions (INT8 and INT4) in addition to measuring memory bandwidth. A composite score is computed based on weighted contributions from each test.

## Features

- **Multi-Precision Tests:**  
  Evaluate performance for:
  - FP32 (5% weight)
  - FP16 (25% weight)
  - FP8 (simulated) (10% weight)
  - INT8 (simulated) (10% weight)
  - INT4 (simulated) (15% weight)

- **Memory Bandwidth Test:**  
  Measures system memory throughput (35% weight).

- **Composite Scoring:**  
  Each metric is normalized against a baseline and combined into a final “Benchmark Points (BP)” score.

- **GUI Interface:**  
  A simple Tkinter-based interface lets you select the target device (CPU, CUDA, or Apple MPS) and view results interactively.

## Requirements

- **Python:** 3.7+
- **PyTorch:** Install via pip:
  ```bash
  pip install torch
  ```
- **Tkinter:** Typically included with Python installations.  
  If not available, install it using your OS package manager.
- **CUDA (Optional):** For running benchmarks on Nvidia GPUs.
- **Apple MPS (Optional):** For benchmarking on Apple Silicon devices.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kl-charizard/llm-benchmark-tool.git
   cd llm-benchmark-tool
   ```

2. **Install dependencies:**

   - If you have a `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```
   - Otherwise, ensure that [PyTorch](https://pytorch.org/) is correctly installed:
     ```bash
     pip install torch
     ```

## Usage

### 1. Start the Benchmark Tool

- **GUI Mode:**  
  Simply run:
  ```bash
  python benchmark.py
  ```
  A window will open with tick boxes for device selection. Select your desired devices and click **"Run Benchmark."**

- **CLI Mode:**  
  Run:
  ```bash
  python benchmark.py -cli
  ```
  The benchmark will run on all available devices (CPU, CUDA if available, MPS if available) and print the results to the terminal.

*Note: This version fixes the memory buffer error and adds a CLI mode as requested.*

### 2. Select the Device

- **Choose your target device:**  
  Select the device (e.g., `cpu`, `cuda`, or `mps`).  
  If you select CUDA or MPS and the corresponding hardware is unavailable, an error message will be displayed.

### 3. Run the Benchmark

- **Click the “Run Benchmark” button:**  
  The tool will perform several tests:
  - **Matrix Multiplication Tests:**  
    For FP32, FP16, simulated FP8, INT8, and INT4 benchmarks.
  - **Memory Bandwidth Test:**  
    Measures the rate of data transfer.
  - **Results:**  
    The performance metrics are logged in the text area, and the overall composite score appears at the bottom.

### 4. Interpreting the Results

- **Matrix Multiplication Tests:**  
  Results are reported in GFLOPS (or GOPS for integer tests).
- **Memory Bandwidth:**  
  Measured in GB/s.
- **Overall Score:**  
  A weighted sum of the normalized performance metrics is provided as the final “Benchmark Points (BP)” score.

## Benchmark Methodology

- **Matrix Multiplication Benchmarks:**  
  The tool performs repeated matrix multiplications on fixed-size matrices (default size: 1024x1024) for several iterations to calculate average performance (GFLOPS/GOPS). Note that simulated quantization is applied for FP8, INT8, and INT4.
  
- **Memory Bandwidth Test:**  
  Measures how fast a large tensor can be cloned, reporting the throughput in GB/s.

- **Composite Score Calculation:**  
  The final score is calculated by normalizing each performance metric against a baseline and weighting it as follows:
  - Memory Bandwidth: 35%
  - FP32: 5%
  - FP16: 25%
  - FP8 (simulated): 10%
  - INT8 (simulated): 10%
  - INT4 (simulated): 15%

## Customization

- **Matrix Size and Iterations:**  
  You can modify the size and iteration parameters in the benchmark functions to suit your testing needs.
  
- **Baseline Values:**  
  Adjust the baseline values in the `BenchmarkTool` class if you have different reference performance metrics.

## Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or additional features, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Disclaimer

This tool is intended for performance evaluation and benchmarking purposes only. Results may vary depending on hardware, system load, and other environmental factors.
