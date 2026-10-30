import os
import urllib.request
import sys

def download_benchmark_video(output_path="data/highway_traffic.mp4"):
    """
    Downloads the official 1080p benchmark highway traffic video (vehicles.mp4)
    from the global research CDN if not already present on disk.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:
        print(f"[INFO] Benchmark video already exists at: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = "https://media.roboflow.com/supervision/video-examples/vehicles.mp4"

    print("=" * 65)
    print("  DOWNLOADING BENCHMARK HIGHWAY TRAFFIC VIDEO (35.3 MB)...")
    print("=" * 65)
    print(f"Source URL: {url}")
    print(f"Target Path: {output_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, (downloaded / total_size) * 100.0)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent:05.1f}% [{mb_down:05.1f} MB / {mb_total:05.1f} MB]")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print("\n[SUCCESS] Benchmark video downloaded successfully!")
        return output_path
    except Exception as e:
        print(f"\n[ERROR] Failed to download benchmark video: {e}")
        return None

if __name__ == "__main__":
    download_benchmark_video()
