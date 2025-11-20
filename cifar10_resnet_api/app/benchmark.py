import time
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
import requests


class FastAPIBenchmark:
    def __init__(self, base_url: str, images_folder: str = "benchmark_images"):
        # Resolve folder relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_folder = os.path.join(script_dir, images_folder)

        print(f"[INFO] Benchmark images folder resolved to: {self.images_folder}")

        self.base_url = base_url.rstrip("/")
        self.images = self._load_images()

    def _load_images(self):
        allowed = (".jpg", ".jpeg", ".png")
        if not os.path.exists(self.images_folder):
            raise Exception(f"Benchmark image folder not found: {self.images_folder}")

        images = [
            os.path.join(self.images_folder, img)
            for img in os.listdir(self.images_folder)
            if img.lower().endswith(allowed)
        ]

        if not images:
            raise Exception("No images found for benchmarking.")

        print(f"[INFO] Loaded {len(images)} benchmark images.")
        return images

    def _send(self, image_path):
        try:
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                start = time.time()
                res = requests.post(f"{self.base_url}/predict", files=files)
                end = time.time()

            latency = (end - start) * 1000

            if res.status_code == 200:
                return {"success": True, "latency": latency}
            else:
                return {"success": False, "latency": latency, "error": res.text}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def latency_test(self, loops=10):
        latencies = []

        for i in range(loops):
            img = self.images[i % len(self.images)]
            result = self._send(img)
            if result["success"]:
                latencies.append(result["latency"])

        if not latencies:
            return {"success": False, "message": "No valid responses received"}

        return {
            "total_requests": loops,
            "successful_requests": len(latencies),
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
        }

    def concurrent_test(self, users=3, duration=5):
        stop_time = time.time() + duration
        latencies = []

        def worker(uid):
            while time.time() < stop_time:
                img = self.images[uid % len(self.images)]
                result = self._send(img)
                if result["success"]:
                    latencies.append(result["latency"])

        with ThreadPoolExecutor(max_workers=users) as executor:
            executor.map(worker, range(users))

        if not latencies:
            return {"success": False, "message": "No valid responses received"}

        return {
            "users": users,
            "duration_seconds": duration,
            "total_requests": len(latencies),
            "requests_per_second": len(latencies) / duration,
            "avg_latency_ms": statistics.mean(latencies),
        }

    def run_all(self):
        return {
            "latency_test": self.latency_test(loops=10),
            "concurrency_test_3_users": self.concurrent_test(users=3, duration=5),
            "concurrency_test_5_users": self.concurrent_test(users=5, duration=5),
        }


if __name__ == "__main__":
    bench = FastAPIBenchmark(
        base_url="http://127.0.0.1:8000",
        images_folder="benchmark_images"
    )
    results = bench.run_all()
    print(results)
