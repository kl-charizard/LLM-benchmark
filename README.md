LLM Benchmark Tool

LLM Benchmark Tool is a Python-based GUI application designed to evaluate the performance of various hardware accelerators—including CPU, CUDA-enabled GPUs, and Apple MPS devices—by testing multiple precision modes. The tool benchmarks floating point (FP32, FP16, simulated FP8) and integer precisions (INT8 and INT4), along with measuring memory bandwidth. It then computes a composite score based on weighted contributions from each test.

Features
	•	Multi-Precision Tests:
Evaluate performance for:
	•	FP32 (5% weight)
	•	FP16 (25% weight)
	•	FP8 (simulated) (10% weight)
	•	INT8 (simulated) (10% weight)
	•	INT4 (simulated) (15% weight)
	•	Memory Bandwidth Test:
Measures system memory throughput (35% weight).
	•	Composite Scoring:
Each metric is normalized against a baseline and combined into a final “Benchmark Points (BP)” score.
	•	GUI Interface:
A simple Tkinter-based interface lets you select the target device (CPU, CUDA, or Apple MPS) and view results interactively.

Requirements
	•	Python 3.7+
	•	PyTorch:
Install via pip:

pip install torch


	•	Tkinter:
Typically included with Python installations. If not, install it using your OS package manager.
	•	CUDA (Optional):
For running benchmarks on Nvidia GPUs.
	•	Apple MPS (Optional):
For benchmarking on Apple Silicon devices.

Installation
	1.	Clone the repository:

git clone https://github.com/yourusername/llm-benchmark-tool.git
cd llm-benchmark-tool


	2.	Install dependencies:
If you have a requirements.txt file:

pip install -r requirements.txt

Otherwise, ensure that PyTorch is installed:

pip install torch



Usage
	1.	Start the Benchmark Tool:


	•	GUI Mode:
Simply run:

python benchmark.py

A window will open with tick boxes for device selection. Select your desired devices and click “Run Benchmark.”

	•	CLI Mode:
Run:

python benchmark.py -cli

The benchmark will run on all available devices (CPU, CUDA if available, MPS if available) and print the results to the terminal.

This version fixes the memory buffer error and adds a CLI mode as requested.


	2.	Select the Device:
	•	Choose your target device (e.g., cpu, cuda, or mps).
	•	Note: If you select CUDA or MPS and the hardware is unavailable, an error message will be displayed.
	3.	Run the Benchmark:
	•	Click the “Run Benchmark” button.
	•	The tool will perform several tests:
	•	Matrix Multiplication Tests: For FP32, FP16, simulated FP8, INT8, and INT4.
	•	Memory Bandwidth Test: Measures the rate of data transfer.
	•	Results will be logged in the text area, and the overall composite score will appear at the bottom.
	4.	Interpreting the Results:
	•	FP32/FP16/FP8/INT8/INT4 Tests: Report performance in GFLOPS (or GOPS for integer tests).
	•	Memory Bandwidth: Measured in GB/s.
	•	Overall Score: A weighted sum of the normalized performance metrics, giving a single “Benchmark Points (BP)” score.

Benchmark Methodology
	•	Matrix Multiplication Benchmarks:
The tool performs repeated matrix multiplications on fixed-size matrices (default: 1024x1024) over several iterations to calculate average performance in GFLOPS/GOPS. For FP8, INT8, and INT4, quantization is simulated.
	•	Memory Bandwidth Test:
A large tensor is repeatedly cloned to measure how quickly data can be transferred in GB/s.
	•	Composite Score Calculation:
Each performance metric is normalized against a baseline value and weighted as follows:
	•	Memory Bandwidth: 35%
	•	FP32: 5%
	•	FP16: 25%
	•	FP8: 10% (simulated)
	•	INT8: 10% (simulated)
	•	INT4: 15% (simulated)

Customization
	•	Matrix Size and Iterations:
Modify the size and iterations parameters in the benchmark functions to suit your testing environment.
	•	Baseline Values:
Update the baseline values in the BenchmarkTool class if you have different reference performance metrics.

Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or additional features, please open an issue or submit a pull request.

License

This project is licensed under the MIT License.

Disclaimer

This tool is intended for performance evaluation and benchmarking purposes only. Results may vary depending on hardware, system load, and other environmental factors.
