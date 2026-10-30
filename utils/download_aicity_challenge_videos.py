import os
import urllib.request
import cv2

AICITY_VIDEOS = [
    {
        "id": "angle1_distant_highway_cctv",
        "name": "Camera Angle 1: Distant Overhead Highway CCTV (Low Resolution & Distant Cars)",
        "path": "videos/HTV/cctv052x2004080516x01638.mp4",
        "challenge": "Tiny distant objects, compression noise, long-range perspective"
    },
    {
        "id": "angle2_dawn_lowlight_intersection",
        "name": "Camera Angle 2: Dawn Low-Light Intersection (Harsh Shadows & High Contrast)",
        "path": "videos/barbados_traffic_challenge/normanniles1/normanniles1_2025-10-20-06-38-45.mp4",
        "challenge": "Low ambient light, long morning shadows, headlight glare"
    },
    {
        "id": "angle3_congested_roundabout",
        "name": "Camera Angle 3: Complex High-Density Roundabout (Curved Trajectories & Turning)",
        "path": "videos/barbados_traffic_challenge/normanniles1/normanniles1_2025-10-20-11-55-45.mp4",
        "challenge": "Dense multi-lane weaving, non-linear trajectories, continuous partial occlusions"
    },
    {
        "id": "angle4_acute_oblique_anomaly",
        "name": "Camera Angle 4: Acute Oblique Angle Junction (Sharp Angle & Sudden Maneuver)",
        "path": "videos/TAD/01_Accident_001.mp4",
        "challenge": "Low viewing angle, sudden lateral acceleration, abrupt speed change"
    },
    {
        "id": "angle5_close_quarters_occlusion",
        "name": "Camera Angle 5: Close-Quarters Roadside View (Heavy Occlusion & Tailgating)",
        "path": "videos/TAD/01_Accident_005.mp4",
        "challenge": "Foreground vehicle blocking background traffic, rapid perspective scaling"
    }
]

def download_and_inspect_aicity():
    base_url = "https://huggingface.co/datasets/Tohrumi/AICityChallenge_Track_3/resolve/main/"
    dest_dir = os.path.join(os.path.dirname(__file__), "..", "data", "aicity_challenge")
    frames_dir = os.path.join(dest_dir, "angle_previews")
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    print("=" * 80)
    print("  DOWNLOADING 5 EXTREME-CHALLENGE CAMERA ANGLES FROM AI CITY CHALLENGE (HF)")
    print("=" * 80)

    downloaded = []
    for idx, v_info in enumerate(AICITY_VIDEOS, 1):
        filename = f"{v_info['id']}.mp4"
        local_path = os.path.join(dest_dir, filename)
        url = base_url + v_info["path"]

        print(f"\n[{idx}/5] Downloading: {v_info['name']}")
        print(f"      Challenge: {v_info['challenge']}")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            with open(local_path, "wb") as f:
                f.write(content)
        
        size_kb = len(content) / 1024.0
        print(f"      Size: {size_kb:.1f} KB -> Saved to: {local_path}")

        # Inspect video
        cap = cv2.VideoCapture(local_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / max(fps, 1.0)

        ret, frame = cap.read()
        if ret:
            preview_path = os.path.join(frames_dir, f"{v_info['id']}_preview.jpg")
            cv2.imwrite(preview_path, frame)
        cap.release()

        print(f"      Resolution: {w}x{h} | FPS: {fps:.1f} | Frames: {total_frames} ({duration:.1f}s)")
        downloaded.append({
            "info": v_info,
            "local_path": local_path,
            "w": w,
            "h": h,
            "fps": fps,
            "duration": duration,
            "preview": preview_path if ret else None
        })

    print("\n" + "=" * 80)
    print(f"  SUCCESSFULLY DOWNLOADED & VERIFIED ALL 5 CHALLENGE ANGLES ({len(downloaded)} videos)")
    print(f"  Location: {dest_dir}")
    print("=" * 80)
    return downloaded

if __name__ == "__main__":
    download_and_inspect_aicity()
