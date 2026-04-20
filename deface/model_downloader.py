"""
Automated model download and verification.

Manages downloading missing ONNX models from a centralized repository
and verifying their integrity via file size checks.
"""

import os
import urllib.request
import urllib.error


# Model registry with download URLs and minimum size validation.
# URLs point to GitHub Releases (v1.0-models tag).
# Format: full direct download URL per model file via GitHub Releases CDN.
# Example: 'https://github.com/fyder9/deface-tw2/releases/download/v1.0-models/scrfd_2.5g.onnx'
MODEL_REGISTRY = {
    'scrfd2.5g': {
        'filename': 'scrfd_2.5g.onnx',
        'url': 'https://github.com/fyder9/deface-tw2/releases/download/v1.0-models/scrfd_2.5g.onnx',
        'min_size_bytes': 3_000_000
    },
    'scrfd10g': {
        'filename': 'scrfd_10g.onnx',
        'url': 'https://github.com/fyder9/deface-tw2/releases/download/v1.0-models/scrfd_10g.onnx',
        'min_size_bytes': 15_000_000
    },
    'crowdhuman': {
        'filename': 'crowdhuman_yolov5m.onnx',
        'url': 'https://github.com/fyder9/deface-tw2/releases/download/v1.0-models/crowdhuman_yolov5m.onnx',
        'min_size_bytes': 75_000_000
    },
    'centerface': {
        'filename': 'centerface.onnx',
        'url': 'https://github.com/fyder9/deface-tw2/releases/download/v1.0-models/centerface.onnx',
        'min_size_bytes': 6_500_000
    },
}


def ensure_model_present(detector_key: str, models_dir: str) -> str:
    """
    Ensure a model file exists and is valid. Download if missing.

    Args:
        detector_key: Key in MODEL_REGISTRY (e.g., 'scrfd2.5g')
        models_dir: Directory to store/look for models

    Returns:
        Path to the model file (absolute path)

    Raises:
        RuntimeError: If model key is unknown, download fails, or file is too small
    """
    # Validate detector key
    if detector_key not in MODEL_REGISTRY:
        known_keys = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise RuntimeError(
            f'Unknown detector key: {detector_key}. '
            f'Known detectors: {known_keys}'
        )

    model_info = MODEL_REGISTRY[detector_key]
    filename = model_info['filename']
    url = model_info['url']
    min_size = model_info['min_size_bytes']

    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    final_path = os.path.join(models_dir, filename)

    # File exists and is large enough — use it
    if os.path.isfile(final_path):
        file_size = os.path.getsize(final_path)
        if file_size >= min_size:
            return final_path
        else:
            # File too small — warn, delete, fall through to download
            print(f'[ModelDownloader] Warning: {filename} exists but is only {file_size} bytes '
                  f'(expected >= {min_size}). Deleting and re-downloading.')
            os.remove(final_path)

    # No URL configured — direct user to manual placement
    if url is None:
        raise RuntimeError(
            f'Model {filename} not found at {final_path}. '
            f'No download URL configured yet. '
            f'Please place the file manually at: {final_path}'
        )

    # Download to temporary path
    tmp_path = final_path + '.tmp'
    try:
        print(f'[ModelDownloader] Downloading {filename}...')
        _download_with_progress(url, tmp_path)

        # Verify downloaded file size
        downloaded_size = os.path.getsize(tmp_path)
        if downloaded_size < min_size:
            os.remove(tmp_path)
            raise RuntimeError(
                f'Downloaded {filename} is only {downloaded_size} bytes '
                f'(expected >= {min_size}). File may be corrupted. '
                f'Download URL: {url}'
            )

        # Move to final location
        os.rename(tmp_path, final_path)
        return final_path

    except Exception as e:
        # Clean up temporary file on any error
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if isinstance(e, RuntimeError):
            raise
        else:
            raise RuntimeError(
                f'Failed to download {filename} from {url}: {str(e)}'
            ) from e


def _download_with_progress(url: str, dest_path: str) -> None:
    """
    Download a file with progress reporting.

    Args:
        url: URL to download from
        dest_path: Path to save file to

    Raises:
        urllib.error.URLError: If download fails
        OSError: If file write fails
    """
    def _progress_hook(block_num, block_size, total_size):
        """Print download progress."""
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        # Cap at total_size to avoid overshoot
        downloaded = min(downloaded, total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f'  {mb_downloaded:.1f} MB / {mb_total:.1f} MB', end='\r')

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook)
    print()  # Newline after progress
