import tkinter as tk
from tkinter import messagebox
import torch
import time
import threading
import sys

class BenchmarkTool:
    def __init__(self, device):
        self.device = device
        # Baseline values for normalization (arbitrarily chosen)
        self.baseline = {
            'mem_bw': 50.0,   # GB/s
            'fp32': 100.0,    # GFLOPS
            'fp16': 200.0,    # GFLOPS
            'fp8': 400.0,     # GFLOPS (simulated)
            'int8': 400.0,    # GOPS (simulated)
            'int4': 800.0     # GOPS (simulated)
        }
        # For high-performance devices (CUDA and MPS), increase workload:
        if self.device.type in ['cuda', 'mps']:
            self.size = 2048                        # Matrix dimension remains 2048
            self.iterations = 60                    # Increased iterations
            self.mem_size = 2 * 1024 * 1024 * 1024    # Set memory test size to 2 GB (in bytes)
        else:
            self.size = 1024
            self.iterations = 10
            self.mem_size = 1024 * 1024 * 10

    def _synchronize(self, tensor):
        # For CUDA use torch.cuda.synchronize(), for MPS force sync by moving result to CPU.
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            _ = tensor.cpu()

    def run_matmul_benchmark(self, dtype):
        """
        Runs a matrix multiplication benchmark.
        For FP32 and FP16, a straightforward matmul is used.
        For FP8, a simulated quantization is applied.
        For INT8 and INT4, integer tensors are used and then cast to int32 for multiplication.
        """
        size = self.size
        iters = self.iterations

        # Floating point benchmarks: FP32 and FP16
        if dtype in [torch.float32, torch.float16]:
            a = torch.randn(size, size, device=self.device, dtype=dtype)
            b = torch.randn(size, size, device=self.device, dtype=dtype)
            # Warmup
            c = torch.matmul(a, b)
            self._synchronize(c)
            start = time.time()
            for _ in range(iters):
                c = torch.matmul(a, b)
                self._synchronize(c)
            end = time.time()
            avg_time = (end - start) / iters
            ops = 2 * (size ** 3)
            gflops = ops / (avg_time * 1e9)
            return gflops

        # Simulated FP8: quantize FP32 numbers to 8-bit (simulate 256 levels) then perform multiplication.
        elif dtype == 'fp8':
            a = torch.randn(size, size, device=self.device, dtype=torch.float32)
            b = torch.randn(size, size, device=self.device, dtype=torch.float32)
            scale = 127.0
            a_q = torch.clamp((a * scale).round(), -128, 127) / scale
            b_q = torch.clamp((b * scale).round(), -128, 127) / scale
            c = torch.matmul(a_q, b_q)
            self._synchronize(c)
            start = time.time()
            for _ in range(iters):
                c = torch.matmul(a_q, b_q)
                self._synchronize(c)
            end = time.time()
            avg_time = (end - start) / iters
            ops = 2 * (size ** 3)
            gflops = ops / (avg_time * 1e9)
            return gflops

        # Simulated INT8: use int8 tensors, then cast to int32 for multiplication.
        elif dtype == 'int8':
            if self.device.type != 'cpu':
                print(f"Skipping INT8 test on {self.device.type} as non-float matrix multiplication is not supported.")
                return 0.0
            a = torch.randint(-128, 127, (size, size), device=self.device, dtype=torch.int8)
            b = torch.randint(-128, 127, (size, size), device=self.device, dtype=torch.int8)
            a_int = a.to(torch.int32)
            b_int = b.to(torch.int32)
            c = torch.matmul(a_int, b_int)
            self._synchronize(c)
            start = time.time()
            for _ in range(iters):
                c = torch.matmul(a_int, b_int)
                self._synchronize(c)
            end = time.time()
            avg_time = (end - start) / iters
            ops = 2 * (size ** 3)
            gops = ops / (avg_time * 1e9)
            return gops

        # Simulated INT4: simulate 4-bit integers (range -8 to 7) then use int32 for multiplication.
        elif dtype == 'int4':
            if self.device.type != 'cpu':
                print(f"Skipping INT4 test on {self.device.type} as non-float matrix multiplication is not supported.")
                return 0.0
            a = torch.randint(-8, 7, (size, size), device=self.device, dtype=torch.int8)
            b = torch.randint(-8, 7, (size, size), device=self.device, dtype=torch.int8)
            a_int = a.to(torch.int32)
            b_int = b.to(torch.int32)
            c = torch.matmul(a_int, b_int)
            self._synchronize(c)
            start = time.time()
            for _ in range(iters):
                c = torch.matmul(a_int, b_int)
                self._synchronize(c)
            end = time.time()
            avg_time = (end - start) / iters
            ops = 2 * (size ** 3)
            gops = ops / (avg_time * 1e9)
            return gops
        else:
            return None

    def run_memory_bandwidth_test(self):
        """
        Measures memory bandwidth by copying a large tensor repeatedly.
        Here we compute the number of float32 elements needed so that the tensor occupies 2GB.
        """
        # For float32, each element is 4 bytes.
        num_elems = self.mem_size // 4
        iters = self.iterations
        a = torch.randn(num_elems, device=self.device)
        self._synchronize(a)
        start = time.time()
        for _ in range(iters):
            b = a.clone()
            self._synchronize(b)
        end = time.time()
        avg_time = (end - start) / iters
        bytes_copied = a.element_size() * a.nelement()
        gb = bytes_copied / (1024 ** 3)
        gbps = gb / avg_time
        return gbps

    def run_all_benchmarks(self):
        results = {}
        results['fp32'] = self.run_matmul_benchmark(torch.float32)
        results['fp16'] = self.run_matmul_benchmark(torch.float16)
        results['fp8'] = self.run_matmul_benchmark('fp8')
        results['int8'] = self.run_matmul_benchmark('int8')
        results['int4'] = self.run_matmul_benchmark('int4')
        results['mem_bw'] = self.run_memory_bandwidth_test()
        return results

    def compute_overall_score(self, results):
        """
        Computes a weighted overall score (Benchmark Points, BP) by normalizing each result
        against its baseline and applying the weight:
          - Memory Bandwidth: 35%
          - FP32: 5%
          - FP16: 25%
          - FP8: 10%
          - INT8: 10%
          - INT4: 15%
        """
        score = (results['mem_bw'] / self.baseline['mem_bw']) * 35
        score += (results['fp32'] / self.baseline['fp32']) * 5
        score += (results['fp16'] / self.baseline['fp16']) * 25
        score += (results['fp8'] / self.baseline['fp8']) * 10
        score += (results['int8'] / self.baseline['int8']) * 10
        score += (results['int4'] / self.baseline['int4']) * 15
        return score

# GUI using Tkinter with tick boxes for device selection
class BenchmarkGUI:
    def __init__(self, root):
        self.root = root
        root.title("LLM Benchmark Tool")

        # Device selection using checkboxes
        tk.Label(root, text="Select Devices:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.cpu_var = tk.IntVar(value=1)  # Default CPU selected
        self.cuda_var = tk.IntVar(value=0)
        self.mps_var = tk.IntVar(value=0)

        self.cpu_cb = tk.Checkbutton(root, text="CPU", variable=self.cpu_var)
        self.cpu_cb.grid(row=1, column=0, padx=10, sticky="w")
        self.cuda_cb = tk.Checkbutton(root, text="CUDA", variable=self.cuda_var)
        self.cuda_cb.grid(row=1, column=1, padx=10, sticky="w")
        self.mps_cb = tk.Checkbutton(root, text="MPS", variable=self.mps_var)
        self.mps_cb.grid(row=1, column=2, padx=10, sticky="w")

        # Button to start benchmark
        self.start_button = tk.Button(root, text="Run Benchmark", command=self.start_benchmark)
        self.start_button.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        # Text area to show log/output
        self.text = tk.Text(root, height=20, width=80)
        self.text.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

    def log(self, message):
        self.text.insert(tk.END, message + "\n")
        self.text.see(tk.END)

    def start_benchmark(self):
        # Disable the start button to prevent multiple concurrent runs.
        self.start_button.config(state=tk.DISABLED)
        self.text.delete(1.0, tk.END)

        # Collect selected devices based on checkbox values.
        devices = []
        if self.cpu_var.get():
            devices.append("cpu")
        if self.cuda_var.get():
            devices.append("cuda")
        if self.mps_var.get():
            devices.append("mps")

        if not devices:
            messagebox.showerror("Error", "Please select at least one device!")
            self.start_button.config(state=tk.NORMAL)
            return

        # Run benchmarks for each selected device in a separate thread.
        threading.Thread(target=self.run_benchmarks, args=(devices,)).start()

    def run_benchmarks(self, devices):
        for device_str in devices:
            # Check if the selected device is available.
            if device_str == "cuda" and not torch.cuda.is_available():
                self.log("CUDA is not available! Skipping.")
                continue
            if device_str == "mps" and not torch.backends.mps.is_available():
                self.log("Apple MPS is not available! Skipping.")
                continue

            device = torch.device(device_str)
            self.log(f"Running benchmark on {device}...")
            benchmark_tool = BenchmarkTool(device)
            results = benchmark_tool.run_all_benchmarks()
            overall_score = benchmark_tool.compute_overall_score(results)
            self.log(f"Results for {device_str}:")
            self.log(f"  FP32 Performance: {results['fp32']:.2f} GFLOPS")
            self.log(f"  FP16 Performance: {results['fp16']:.2f} GFLOPS")
            self.log(f"  FP8 (simulated) Performance: {results['fp8']:.2f} GFLOPS")
            self.log(f"  INT8 (simulated) Performance: {results['int8']:.2f} GOPS")
            self.log(f"  INT4 (simulated) Performance: {results['int4']:.2f} GOPS")
            self.log(f"  Memory Bandwidth: {results['mem_bw']:.2f} GB/s")
            self.log(f"  Overall Score: {overall_score:.2f} Benchmark Points (BP)")
            self.log("-" * 60)
        self.start_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    if "-cli" in sys.argv:
        # CLI mode: Run benchmarks on all available devices and print results.
        available_devices = []
        available_devices.append("cpu")
        if torch.cuda.is_available():
            available_devices.append("cuda")
        if torch.backends.mps.is_available():
            available_devices.append("mps")
        for device_str in available_devices:
            device = torch.device(device_str)
            print(f"Running benchmark on {device}...")
            benchmark_tool = BenchmarkTool(device)
            results = benchmark_tool.run_all_benchmarks()
            overall_score = benchmark_tool.compute_overall_score(results)
            print(f"Results for {device_str}:")
            print(f"  FP32 Performance: {results['fp32']:.2f} GFLOPS")
            print(f"  FP16 Performance: {results['fp16']:.2f} GFLOPS")
            print(f"  FP8 (simulated) Performance: {results['fp8']:.2f} GFLOPS")
            print(f"  INT8 (simulated) Performance: {results['int8']:.2f} GOPS")
            print(f"  INT4 (simulated) Performance: {results['int4']:.2f} GOPS")
            print(f"  Memory Bandwidth: {results['mem_bw']:.2f} GB/s")
            print(f"  Overall Score: {overall_score:.2f} Benchmark Points (BP)")
            print("-" * 60)
    else:
        # GUI mode
        root = tk.Tk()
        app = BenchmarkGUI(root)
        root.mainloop()
