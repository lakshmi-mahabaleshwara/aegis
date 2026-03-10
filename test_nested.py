import os
from PIL import Image
import numpy as np
import subprocess

# 1. Clear staging_output
os.system('rm -rf staging_output/*')

# 2. Create nested structure
target_dir = 'staging_input/deeply/nested/ultrasound_sweep'
os.makedirs(target_dir, exist_ok=True)

img = np.zeros((512, 512, 3), dtype=np.uint8)
img[50:100, 50:100] = 255
Image.fromarray(img).save(os.path.join(target_dir, 'img_A.png'))
Image.fromarray(img).save(os.path.join(target_dir, 'img_B.png'))

print("Created dummy images in:", target_dir)

# 3. Run pipeline
print("Running pipeline...")
env = os.environ.copy()
env['PYTHONPATH'] = 'monai_aegis'
result = subprocess.run(
    ['.venv/bin/python', 'run_image_pipeline.py', '--config', 'monai_aegis/config/config.yaml'],
    env=env,
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("ERRORS:", result.stderr)
