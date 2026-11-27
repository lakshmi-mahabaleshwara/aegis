"""Behavior fingerprint for the ``monai_aegis.transforms.metadata`` refactor.

Runs a fixed set of 36 scrub scenarios (header-only, every pixel-injection
layout, error paths, and all three ``ScrubDicomMetadatad`` modes) with a
fixed salt and writes a JSON fingerprint: SHA-256 of each serialized output
dataset, the full audit-row lists, tensor shapes/dtypes/hashes, and error
messages. Output is deterministic, so a refactor is behavior-preserving iff
the file produced from the refactored tree is byte-identical to the one
produced from the baseline tree::

    python scripts/metadata_refactor_fingerprint.py <baseline>/src  /tmp/base.json
    python scripts/metadata_refactor_fingerprint.py <refactor>/src  /tmp/new.json
    cmp /tmp/base.json /tmp/new.json && echo behavior preserved

Complements tests/unit/test_metadata_characterization.py (which asserts
the same behaviors as readable unit tests); this script catches anything
those assertions do not spell out, down to the byte level.
Requires the full runtime (pydicom, numpy, torch, monai).
"""
import sys, json, hashlib, io, os, logging
sys.path.insert(0, sys.argv[1])
logging.disable(logging.CRITICAL)
import numpy as np, pydicom, pydicom.uid, torch
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from monai.data import MetaTensor
from monai_aegis.transforms.metadata import ScrubDicomMetadata, ScrubDicomMetadatad
from monai_aegis.transforms.exceptions import MetadataScrubError

F = "1.2.840.113619.2.55."
STD = "1.2.840.10008"

def ds_bytes(ds):
    b = io.BytesIO()
    ds.is_little_endian = True; ds.is_implicit_VR = False
    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = getattr(ds.file_meta, "TransferSyntaxUID", pydicom.uid.ExplicitVRLittleEndian)
    pydicom.dcmwrite(b, ds, write_like_original=True)
    return hashlib.sha256(b.getvalue()).hexdigest()

def rich_ds(i=0, rows=4, cols=6, rgb=False, frames=None):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = STD + ".5.1.4.1.1.7"
    ds.file_meta.MediaStorageSOPInstanceUID = F + f"sop.{i}"
    ds.file_meta.ImplementationClassUID = F + "impl"
    ds.SOPClassUID = STD + ".5.1.4.1.1.7"; ds.SOPInstanceUID = F + f"sop.{i}"
    ds.StudyInstanceUID = F + "study"; ds.SeriesInstanceUID = F + "series"; ds.FrameOfReferenceUID = F + "for"
    ds.PatientName = "Doe^Jane"; ds.PatientID = "MRN-42"; ds.PatientBirthDate = "19700101"
    ds.StudyDate = "20240101"; ds.AccessionNumber = "ACC-1"; ds.StudyID = "788962607"
    ds.InstitutionName = "General Hospital"; ds.Modality = "CT"; ds.BurnedInAnnotation = "YES"
    ds.add_new((0x0008, 0x010C), "UI", STD + ".2.16.4")
    ds.add_new((0x0008, 0x1155), "UI", [F + "study", STD + ".1.2"])
    inner = Dataset(); inner.PatientID = "INNER"; inner.ReferencedSOPInstanceUID = F + "study"
    inner.ReferencedSOPClassUID = STD + ".5.1.4.1.1.7"
    ib = inner.private_block(0x0011, "INNER PRIV", 0x10); ib.add_new(0x01, "LO", "x")
    mid = Dataset(); mid.PatientName = "Mid^Name"; mid.StudyDate = "20240102"
    mid.add_new((0x0008, 0x1140), "SQ", Sequence([inner]))
    ds.add_new((0x0040, 0x0275), "SQ", Sequence([mid]))
    pb = ds.private_block(0x0009, "TOP PRIV", 0x10); pb.add_new(0x01, "LO", "secret")
    ds.Rows, ds.Columns = rows, cols
    ds.PixelSpacing = [0.5, 0.5]; ds.SliceThickness = 2.0; ds.SpacingBetweenSlices = 2.0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; ds.ImagePositionPatient = [0, 0, float(i)]
    ds.LargestImagePixelValue = 4000; ds.WindowCenter = "999"; ds.WindowWidth = "999"
    if rgb:
        ds.SamplesPerPixel = 3; ds.PhotometricInterpretation = "RGB"; ds.PlanarConfiguration = 0
        ds.BitsAllocated = ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0
        shape = (frames, rows, cols, 3) if frames else (rows, cols, 3)
        ds.PixelData = (np.arange(np.prod(shape)) % 251).astype(np.uint8).reshape(shape).tobytes()
    else:
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
        shape = (frames, rows, cols) if frames else (rows, cols)
        ds.PixelData = (np.arange(np.prod(shape)) % 4001).astype(np.uint16).reshape(shape).tobytes()
    if frames: ds.NumberOfFrames = str(frames)
    return ds

MAPPING = {"(0010,0010)": "DUMMY", "(0010,0020)": "DUMMY", "(0010,0030)": "REMOVE", "(0008,0020)": "ZERO",
           "(0008,0050)": "REMOVE", "(0020,0010)": "DUMMY", "(0008,0080)": "KEEP", "(0028,1050)": "REMOVE",
           "(0x0040,0x0275)": "KEEP"}
CFG = {"pii_mapping": MAPPING, "tokenization": {"salt": "golden-salt"}}
out = {}

# --- array transform: header only, pixel variants, ordering
t = ScrubDicomMetadata(CFG)
cases = {
    "header_only": (rich_ds(), None),
    "px_2d_u16": (rich_ds(), (np.arange(24) % 4001).astype(np.uint16).reshape(4, 6) * 1),
    "px_2d_i16": (rich_ds(), (np.arange(24).astype(np.int16) - 12).reshape(4, 6)),
    "px_2d_u8": (rich_ds(), np.arange(24, dtype=np.uint8).reshape(4, 6)),
    "px_hw3": (rich_ds(), np.arange(72, dtype=np.uint8).reshape(4, 6, 3)),
    "px_hw4": (rich_ds(), np.arange(96, dtype=np.uint8).reshape(4, 6, 4)),
    "px_dhw": (rich_ds(), (np.arange(120) % 4001).astype(np.uint16).reshape(5, 4, 6)),
    "px_dhw3": (rich_ds(), (np.arange(360) % 251).astype(np.uint8).reshape(5, 4, 6, 3)),
    "px_dhw4": (rich_ds(), (np.arange(480) % 251).astype(np.uint8).reshape(5, 4, 6, 4)),
    "no_sop_uid": (Dataset(), None),
}
for name, (ds, px) in cases.items():
    o, rows = t.scrub(f"/in/{name}.dcm", pixel_data=px, dataset=ds)
    out[name] = {"ds": ds_bytes(o), "rows": rows, "fm": str(o.file_meta) if hasattr(o, "file_meta") else None}
for shape in ((5,), (5, 4, 6, 1), (2, 5, 4, 6, 3)):
    try:
        t.scrub("x", pixel_data=np.zeros(shape, dtype=np.uint8), dataset=rich_ds()); out[f"err_{shape}"] = "no error"
    except MetadataScrubError as e:
        out[f"err_{shape}"] = [str(e), e.transform, e.uri]
# remap_uid API
out["remap_uid"] = [t.remap_uid(F + "z"), t.remap_uid("", fallback_entropy="/p/x.dcm"), t.remap_uid(" " + F + "z ")]
out["call_returns_dataset"] = ds_bytes(t(uri="x", dataset=rich_ds()))
# construction-time validation messages
for bad in ({"(0010,0010)": "DELETE"}, {"(0010)": "REMOVE"}, {"zz": "REMOVE"}):
    try: ScrubDicomMetadata({"pii_mapping": bad}); out[f"cfg_{bad}"] = "no error"
    except ValueError as e: out[f"cfg_{bad}"] = str(e)

# --- dict transform
def run_dict(name, d):
    tr = ScrubDicomMetadatad(keys=["image"], config=CFG)
    try:
        r = tr(d)
    except MetadataScrubError as e:
        out[name] = ["ERR", str(e), e.transform, e.uri]; return
    rec = {"keys": sorted(str(k) for k in r)}
    img = r["image"]
    arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    rec["image"] = [type(img).__name__, list(arr.shape), str(arr.dtype), hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()]
    if hasattr(img, "meta"): rec["meta"] = {k: str(v) for k, v in img.meta.items() if k != "affine"}
    if "image_scrubbed_ds" in r: rec["scrubbed_ds"] = ds_bytes(r["image_scrubbed_ds"]); rec["rows"] = r["image_tag_actions"]
    if "image_scrubbed_datasets" in r:
        rec["scrubbed_datasets"] = [ds_bytes(s) for s in r["image_scrubbed_datasets"]]; rec["rows_per_slice"] = r["image_tag_actions_per_slice"]
    out[name] = rec

md = lambda **kw: {"filename_or_obj": "/in/a.dcm", **kw}
run_dict("d_single_norm", {"image": MetaTensor(torch.full((1, 4, 6), 0.5), meta=md()), "image_meta_dict": md(), "image_dicom_dataset": rich_ds()})
run_dict("d_single_raw", {"image": MetaTensor(torch.arange(24, dtype=torch.float32).reshape(1, 4, 6) * 10, meta=md()), "image_meta_dict": md(), "image_dicom_dataset": rich_ds()})
run_dict("d_single_nd", {"image": np.full((1, 4, 6), 500.0, dtype=np.float32), "image_meta_dict": md(), "image_dicom_dataset": rich_ds()})
run_dict("d_single_2d", {"image": np.full((4, 6), 500.0, dtype=np.float32), "image_meta_dict": md(), "image_dicom_dataset": rich_ds()})
run_dict("d_single_rgb", {"image": MetaTensor(torch.arange(72, dtype=torch.float32).reshape(3, 4, 6), meta=md()), "image_meta_dict": md(), "image_dicom_dataset": rich_ds(rgb=True)})
run_dict("d_single_list_fpath", {"image": MetaTensor(torch.full((1, 4, 6), 500.0)), "image_meta_dict": {"filename_or_obj": ["/in/a.dcm", "/in/b.dcm"]}, "image_dicom_dataset": rich_ds()})
run_dict("d_skip_nondicom", {"image": MetaTensor(torch.zeros((3, 4, 6))), "image_meta_dict": {"filename_or_obj": "/in/a.jpg"}})
run_dict("d_skip_nofname", {"image": MetaTensor(torch.zeros((1, 4, 6))), "image_meta_dict": {}, "image_dicom_dataset": rich_ds()})
n = 3
smd = {"filename_or_obj": "/in/s0.dcm", "is_multiframe": False, "slice_uris": [f"/in/s{i}.dcm" for i in range(n)]}
run_dict("d_series", {"image": MetaTensor(torch.full((1, n, 4, 6), 0.25), meta=dict(smd)), "image_meta_dict": dict(smd), "image_dicom_datasets": [rich_ds(i) for i in range(n)]})
run_dict("d_series_raw_nd", {"image": np.arange(n * 24, dtype=np.float32).reshape(1, n, 4, 6) * 7, "image_meta_dict": dict(smd), "image_dicom_datasets": [rich_ds(i) for i in range(n)]})
run_dict("d_series_rgb", {"image": MetaTensor(torch.arange(3 * n * 24, dtype=torch.float32).reshape(3, n, 4, 6) % 251), "image_meta_dict": dict(smd), "image_dicom_datasets": [rich_ds(i, rgb=True) for i in range(n)]})
run_dict("d_series_short_uris", {"image": MetaTensor(torch.full((1, n, 4, 6), 0.25)), "image_meta_dict": {**smd, "slice_uris": ["/in/only.dcm"]}, "image_dicom_datasets": [rich_ds(i) for i in range(n)]})
run_dict("d_series_fallback_3d", {"image": MetaTensor(torch.full((1, 4, 6), 500.0)), "image_meta_dict": dict(smd), "image_dicom_datasets": [rich_ds(0)], "image_dicom_dataset": rich_ds(0)})
mmd = {"filename_or_obj": "/in/mf.dcm", "is_multiframe": True, "slice_uris": ["/in/mf.dcm"]}
run_dict("d_mf", {"image": MetaTensor(torch.full((1, 3, 4, 6), 0.5), meta=dict(mmd)), "image_meta_dict": dict(mmd), "image_dicom_datasets": [rich_ds(frames=3)]})
run_dict("d_mf_raw", {"image": MetaTensor(torch.arange(72, dtype=torch.float32).reshape(1, 3, 4, 6) * 50), "image_meta_dict": dict(mmd), "image_dicom_datasets": [rich_ds(frames=3)]})
run_dict("d_mf_rgb", {"image": MetaTensor(torch.arange(3 * 72, dtype=torch.float32).reshape(3, 3, 4, 6) % 251), "image_meta_dict": dict(mmd), "image_dicom_datasets": [rich_ds(rgb=True, frames=3)]})
run_dict("d_mf_3d_err", {"image": MetaTensor(torch.zeros((1, 4, 6))), "image_meta_dict": dict(mmd), "image_dicom_datasets": [rich_ds(frames=3)]})
run_dict("d_mf_chan_err", {"image": MetaTensor(torch.zeros((2, 3, 4, 6))), "image_meta_dict": dict(mmd), "image_dicom_datasets": [rich_ds(frames=3)]})

json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True, default=str)
print("cases:", len(out), "->", sys.argv[2])
