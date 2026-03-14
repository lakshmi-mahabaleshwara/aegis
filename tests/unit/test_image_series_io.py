"""Unit tests for image series I/O transforms."""
import os
import tempfile
import unittest

from PIL import Image

from monai_aegis.transforms.image_series_io import LoadImageSeriesd


class TestLoadImageSeriesd(unittest.TestCase):
    """Test tokenization behavior for image series loaders."""

    def test_target_token_uses_top_level_relative_folder_for_series(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            series_dir = os.path.join(input_dir, "patient_a", "series_1")
            os.makedirs(series_dir)
            paths = []
            for i in range(2):
                fp = os.path.join(series_dir, f"slice_{i}.png")
                Image.new("RGB", (10, 10), color="red").save(fp)
                paths.append(fp)

            loader = LoadImageSeriesd(
                keys=["image"],
                config={"tokenization": {"salt": "test-salt"}},
                input_dir=input_dir,
            )
            data = loader({"image": paths})

            self.assertIsNotNone(data["image_target_token"])

    def test_target_token_is_none_for_root_level_series(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            os.makedirs(input_dir)
            paths = []
            for i in range(2):
                fp = os.path.join(input_dir, f"slice_{i}.png")
                Image.new("RGB", (10, 10), color="red").save(fp)
                paths.append(fp)

            loader = LoadImageSeriesd(
                keys=["image"],
                config={"tokenization": {"salt": "test-salt"}},
                input_dir=input_dir,
            )
            data = loader({"image": paths})

            self.assertIsNone(data["image_target_token"])


if __name__ == "__main__":
    unittest.main()
