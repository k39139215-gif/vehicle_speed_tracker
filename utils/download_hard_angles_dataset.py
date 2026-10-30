import os
import sys
import json
import time
import urllib.request
import zlib
import cv2

tracker_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if tracker_dir not in sys.path:
    sys.path.append(tracker_dir)

from core.vehicle_detector import VehicleDetector
from core.vehicle_tracker import VehicleTracker

DATA_DIR = os.path.join(tracker_dir, 'data', 'new_hard_angles')
UAVDT_GT_DIR = os.path.join(DATA_DIR, 'uavdt_gt')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UAVDT_GT_DIR, exist_ok=True)

CHALLENGE_VIDEOS = [
    {
        'id': 'video1_tunnel_surveillance',
        'title': 'Angle 1: Enclosed Subterranean Roadway Tunnel (Reflective Sodium Lighting)',
        'hf_path': 'Accident-Bench/land_space/long/videos/000004.mp4',
        'speed_limit': 60.0,
        'challenge': 'Severe artificial sodium glare, metallic reverberation, tight multi-lane corridor, dark occlusion'
    },
    {
        'id': 'video2_elevated_night_intersection',
        'title': 'Angle 2: Elevated High-Mast 02:00 AM Nocturnal Intersection (Long Shadows & Glare)',
        'hf_path': 'Accident-Bench/land_space/long/videos/000015.mp4',
        'speed_limit': 50.0,
        'challenge': 'Steep high-mast camera angle, high-contrast light blooming, pedestrian & cyclist blindspots'
    },
    {
        'id': 'video3_blizzard_snow_roadway',
        'title': 'Angle 3: Severe Blizzard & Snowstorm Highway Overpass (Whiteout & Spray)',
        'hf_path': 'Accident-Bench/land_space/long/videos/000021.mp4',
        'speed_limit': 45.0,
        'challenge': 'Falling snow particles, tire slush plumes, low visual contrast, obscured lane markings'
    },
    {
        'id': 'video4_zebra_crossing_tjunction',
        'title': 'Angle 4: Multi-Branch Urban T-Junction & Zebra Crossing (Complex Turn Maneuvers)',
        'hf_path': 'Accident-Bench/land_space/long/videos/000005.mp4',
        'speed_limit': 40.0,
        'challenge': 'Dense mixed traffic, stationary 2-wheelers at pedestrian crossing, abrupt transverse braking'
    },
    {
        'id': 'video5_curved_highway_overpass',
        'title': 'Angle 5: Acute Curved Highway Overpass & On-Ramp (Rapid Perspective Shift)',
        'hf_path': 'Accident-Bench/land_space/long/videos/000022.mp4',
        'speed_limit': 70.0,
        'challenge': 'Non-linear road curvature, high-speed merging vehicles, extreme foreground-to-background scaling'
    }
]

def download_file(url, out_path):
    print(f'  Downloading: {os.path.basename(out_path)}...')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(out_path, 'wb') as f:
        total = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        chunk_size = 128 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100
                print(f'\r    Progress: {downloaded/1024/1024:.2f} MB / {total/1024/1024:.2f} MB ({pct:.1f}%)', end='', flush=True)
    print('\n    Done!')

def main():
    print('=' * 80)
    print('  DOWNLOADING 5 NEW HARD/DIFFICULT CAMERA ANGLES + GROUND TRUTH ANNOTATIONS')
    print('=' * 80)

    base_video_url = 'https://huggingface.co/datasets/Tohrumi/AICityChallenge_Track_3/resolve/main/videos/'
    annotation_types = ['scene_description', 'temporal_localization', 'temporal_description', 'causal_linkage']

    # 1. Fetch JSON Annotations for all 5 videos
    print('\n[STEP 1/4] Fetching official challenge metadata & ground-truth JSON annotations...')
    gt_cache = {v['id']: {} for v in CHALLENGE_VIDEOS}

    for atype in annotation_types:
        url = f'https://huggingface.co/datasets/Tohrumi/AICityChallenge_Track_3/raw/main/train/{atype}.json'
        print(f'  Fetching {atype}.json from Hugging Face...')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for item in data.get('items', []):
                vid = item.get('video_id')
                for c in CHALLENGE_VIDEOS:
                    if vid == c['hf_path']:
                        gt_cache[c['id']][atype] = item

    # Save individual JSON annotation files
    for c in CHALLENGE_VIDEOS:
        gt_file = os.path.join(DATA_DIR, f"{c['id']}_annotations.json")
        with open(gt_file, 'w', encoding='utf-8') as f:
            json.dump({
                'video_id': c['id'],
                'title': c['title'],
                'speed_limit_kmh': c['speed_limit'],
                'challenge_description': c['challenge'],
                'official_ground_truth': gt_cache[c['id']]
            }, f, indent=2)
        print(f'  -> Saved: {os.path.basename(gt_file)}')

    # 2. Download the 5 Video Files
    print('\n[STEP 2/4] Downloading 5 extreme-difficulty challenge videos...')
    for idx, c in enumerate(CHALLENGE_VIDEOS, 1):
        v_url = base_video_url + c['hf_path']
        v_dest = os.path.join(DATA_DIR, f"{c['id']}.mp4")
        if not os.path.exists(v_dest) or os.path.getsize(v_dest) == 0:
            print(f'[{idx}/5] {c["title"]}')
            download_file(v_url, v_dest)
        else:
            print(f'[{idx}/5] [CACHED] {c["id"]}.mp4 ({os.path.getsize(v_dest)/1024/1024:.2f} MB)')

    # 3. Extract UAVDT Official MOT Ground-Truth Tracking Files
    print('\n[STEP 3/4] Extracting official MOT tracking benchmark annotations from UAVDT...')
    uavdt_url = 'https://huggingface.co/datasets/vanthanh/UAVDT-Benchmark-M/resolve/main/UAV-benchmark-MOTD_v1.0.zip'
    total_size = 245719325

    # Read central directory
    req_tail = urllib.request.Request(uavdt_url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={total_size-131072}-{total_size-1}'})
    with urllib.request.urlopen(req_tail) as resp:
        tail = resp.read()
        eocd_pos = tail.rfind(b'PK\x05\x06')
        eocd = tail[eocd_pos:]
        cd_size = int.from_bytes(eocd[12:16], 'little')
        cd_offset = int.from_bytes(eocd[16:20], 'little')

    req_cd = urllib.request.Request(uavdt_url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={cd_offset}-{cd_offset+cd_size-1}'})
    with urllib.request.urlopen(req_cd) as resp_cd:
        cd_data = resp_cd.read()

    uav_targets = {
        'UAV-benchmark-MOTD_v1.0/GT/M0701_gt.txt': 'M0701_heavy_fog_high_alt_birdview_gt.txt',
        'UAV-benchmark-MOTD_v1.0/GT/M0203_gt.txt': 'M0203_pitch_black_night_birdview_gt.txt',
        'UAV-benchmark-MOTD_v1.0/GT/M0601_gt.txt': 'M0601_high_alt_oblique_overpass_gt.txt',
        'UAV-benchmark-MOTD_v1.0/GT/M1101_gt.txt': 'M1101_night_roadside_acute_oblique_gt.txt',
        'UAV-benchmark-MOTD_v1.0/GT/M0205_gt.txt': 'M0205_night_lateral_profile_view_gt.txt'
    }

    pos = 0
    while pos < len(cd_data) and cd_data[pos:pos+4] == b'PK\x01\x02':
        fn_len = int.from_bytes(cd_data[pos+28:pos+30], 'little')
        extra_len = int.from_bytes(cd_data[pos+30:pos+32], 'little')
        comment_len = int.from_bytes(cd_data[pos+32:pos+34], 'little')
        fname = cd_data[pos+46:pos+46+fn_len].decode('utf-8', errors='ignore')

        if fname in uav_targets:
            comp_size = int.from_bytes(cd_data[pos+20:pos+24], 'little')
            local_offset = int.from_bytes(cd_data[pos+42:pos+46], 'little')
            save_name = uav_targets[fname]
            save_path = os.path.join(UAVDT_GT_DIR, save_name)

            req_f = urllib.request.Request(uavdt_url, headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={local_offset}-{local_offset+comp_size+200}'})
            with urllib.request.urlopen(req_f) as resp_f:
                cdata_chunk = resp_f.read()
                l_fn = int.from_bytes(cdata_chunk[26:28], 'little')
                l_ex = int.from_bytes(cdata_chunk[28:30], 'little')
                raw_compressed = cdata_chunk[30+l_fn+l_ex:30+l_fn+l_ex+comp_size]
                decomp = zlib.decompress(raw_compressed, -15)
                with open(save_path, 'wb') as sf:
                    sf.write(decomp)
                print(f'  -> Extracted MOT Benchmark GT: {save_name} ({len(decomp.splitlines())} lines)')

        pos += 46 + fn_len + extra_len + comment_len

    # 4. Generate Frame-Level Vehicle Tracking Annotations for the 5 Videos
    print('\n[STEP 4/4] Generating frame-level tracking & bounding box annotations (YOLOv8 + ByteTrack)...')
    detector = VehicleDetector(conf_thresh=0.25)
    tracker = VehicleTracker(max_distance_threshold=65.0, max_lost_frames=15)

    for c in CHALLENGE_VIDEOS:
        vpath = os.path.join(DATA_DIR, f"{c['id']}.mp4")
        csv_path = os.path.join(DATA_DIR, f"{c['id']}_frame_tracking.csv")

        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f'  Annotating {c["id"]} ({w}x{h}, {total_frames} frames)...')
        lines = ['frame_idx,timestamp_sec,track_id,class_name,x1,y1,x2,y2']

        # Process first 100 frames
        f_idx = 0
        while cap.isOpened() and f_idx < min(100, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            t_sec = f_idx / fps
            dets = detector.detect(frame)
            tracks = tracker.update(dets, timestamp=t_sec)

            for tid, trk in tracks.items():
                box = trk.bbox
                lines.append(f"{f_idx},{t_sec:.2f},{tid},{trk.class_name},{box[0]},{box[1]},{box[2]},{box[3]}")
            f_idx += 1

        cap.release()
        with open(csv_path, 'w', encoding='utf-8') as cf:
            cf.write('\n'.join(lines))
        print(f'    -> Saved {len(lines)-1} detection records to {os.path.basename(csv_path)}')

    print('\n' + '=' * 80)
    print('  SUCCESSFULLY DOWNLOADED ALL 5 EXTREME-CHALLENGE ANGLES & ANNOTATIONS!')
    print(f'  Location: {DATA_DIR}')
    print('=' * 80)

if __name__ == '__main__':
    main()
