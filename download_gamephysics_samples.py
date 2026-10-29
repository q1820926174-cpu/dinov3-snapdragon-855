"""Download four public GamePhysics archives and extract traceable real frames.

These gameplay-bug clips are a distribution-biased alternative to Roboflow.
No bounding-box labels are provided. Source video and timestamp are retained.
"""
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path
import imageio_ffmpeg
from download_hf_samples import get

REPO = 'asgaardlab/GamePhysics'
GAMES = {'pvz': 'plants_vs_zombies', 'hollow_knight': 'hollow_knight',
         'cuphead': 'cuphead', 'clash_royale': 'clash_royale'}


def main():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cache = Path('downloads/gamephysics')
    cache.mkdir(parents=True, exist_ok=True)
    for game, archive in GAMES.items():
        url = f'https://huggingface.co/datasets/{REPO}/resolve/main/data/{archive}.tar.gz'
        package = cache / (archive+'.tar.gz')
        if not package.exists():
            package.write_bytes(get(url))
        videos = []
        with tarfile.open(package) as tar:
            for member in sorted(tar.getmembers(), key=lambda x: x.name):
                if not member.isfile() or not member.name.endswith('.mp4'):
                    continue
                # Copy only media bytes to a controlled basename; no tar path extraction.
                video = cache / game / Path(member.name).name
                video.parent.mkdir(parents=True, exist_ok=True)
                if not video.exists():
                    video.write_bytes(tar.extractfile(member).read())
                result = subprocess.run([ffmpeg, '-hide_banner', '-i', str(video)], capture_output=True, text=True)
                match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', result.stderr)
                if not match:
                    raise RuntimeError(f'Cannot read duration: {video}')
                duration = sum(float(t)*factor for t, factor in zip(match.groups(), (3600,60,1)))
                videos.append((video, duration))
        # Largest video is held out; all remaining recordings are calibration-only.
        test_video = max(videos, key=lambda item: item[1])[0] if len(videos)>1 else None
        manifest_path = Path('datasets_raw')/game/'source_manifest.json'
        records = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        for video, duration in videos:
            count = min(256, max(1, int(duration*2)))
            for i in range(count):
                timestamp = (i+0.5)*duration/count
                if test_video is None:
                    # Single-video PvZ: temporal split with a 2-second guard band.
                    boundary = duration*0.7
                    if abs(timestamp-boundary)<2:
                        continue
                    split = 'train' if timestamp<boundary else 'test'
                else:
                    split = 'test' if video == test_video else 'train'
                path = Path('datasets_raw')/game/split/f'{video.stem}_ms{round(timestamp*1000):07d}.jpg'
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-ss', str(timestamp),
                                    '-i', str(video), '-frames:v', '1', '-q:v', '2', '-y', str(path)], check=True)
                if not path.exists():
                    continue
                records.append({'file': path.relative_to(Path('datasets_raw')/game).as_posix(),
                                'dataset': REPO, 'url': url, 'video_id': video.stem,
                                'timestamp_seconds': timestamp, 'source_split': split,
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'archive_sha256': hashlib.sha256(package.read_bytes()).hexdigest()})
            print(game, video.stem, duration, 'frames', len(records), flush=True)
        records = list({r['file']:r for r in records}.values())
        manifest_path.write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
