import os
import io
import urllib.request
import zipfile
import time

class BufferedHTTPRangeFile(io.RawIOBase):
    """
    Buffered HTTP Range File wrapper for zipfile.ZipFile.
    Enables random access and reading of remote ZIP archives without
    downloading the entire multi-gigabyte archive.
    """
    def __init__(self, url, buffer_size=4 * 1024 * 1024):
        self.url = url
        self.buffer_size = buffer_size
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}, method='HEAD')
        with urllib.request.urlopen(req, timeout=15) as resp:
            self.size = int(resp.headers['Content-Length'])
            
        self.pos = 0
        self.buf_start = -1
        self.buf_end = -1
        self.buffer = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        l = len(b)
        if self.pos >= self.size or l == 0:
            return 0

        # Check if requested range is fully within cache
        if self.buf_start <= self.pos and (self.pos + l) <= self.buf_end:
            offset = self.pos - self.buf_start
            b[:l] = self.buffer[offset:offset+l]
            self.pos += l
            return l

        # Cache miss: fetch a larger chunk (buffer_size) starting at self.pos
        fetch_len = max(l, self.buffer_size)
        end = min(self.pos + fetch_len - 1, self.size - 1)
        
        req = urllib.request.Request(
            self.url,
            headers={'Range': f'bytes={self.pos}-{end}', 'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            self.buffer = resp.read()
            self.buf_start = self.pos
            self.buf_end = self.pos + len(self.buffer)

        actual_len = min(l, len(self.buffer))
        b[:actual_len] = self.buffer[:actual_len]
        self.pos += actual_len
        return actual_len


class VS13DataLoader:
    """
    Manages selective download and streaming from the official Slobodan VS13 repository:
    http://slobodan.ucg.ac.me/science/vs13/
    """
    BASE_URL = "http://slobodan.ucg.ac.me/science/vs13/"
    
    VEHICLES = [
        "CitroenC4Picasso",
        "KiaSportage",
        "Mazda3",
        "Peugeot208",
        "VWPassat"
    ]

    def __init__(self, cache_dir="downloads"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._zip_handles = {}

    def get_audio_zip(self, vehicle_name):
        if vehicle_name not in self._zip_handles:
            url = f"{self.BASE_URL}audio+annotations/{vehicle_name}.zip"
            range_f = BufferedHTTPRangeFile(url)
            zf = zipfile.ZipFile(range_f)
            self._zip_handles[vehicle_name] = zf
        return self._zip_handles[vehicle_name]

    def list_recordings(self, vehicle_name):
        zf = self.get_audio_zip(vehicle_name)
        txt_files = [f for f in zf.namelist() if f.endswith(".txt") and "/" in f]
        recs = [os.path.splitext(os.path.basename(f))[0] for f in txt_files]
        return sorted(recs)

    def fetch_annotation(self, vehicle_name, recording_id):
        zf = self.get_audio_zip(vehicle_name)
        txt_path = f"{vehicle_name}/{recording_id}.txt"
        content = zf.read(txt_path).decode("utf-8").strip()
        parts = content.split()
        speed_kmh = float(parts[0])
        cpa_sec = float(parts[1]) if len(parts) > 1 else 0.0
        return speed_kmh, cpa_sec

    def fetch_audio_file(self, vehicle_name, recording_id):
        zf = self.get_audio_zip(vehicle_name)
        wav_path = f"{vehicle_name}/{recording_id}.wav"
        data = zf.read(wav_path)
        
        target_dir = os.path.join(self.cache_dir, vehicle_name)
        os.makedirs(target_dir, exist_ok=True)
        local_path = os.path.join(target_dir, f"{recording_id}.wav")
        with open(local_path, "wb") as f:
            f.write(data)
        return local_path
