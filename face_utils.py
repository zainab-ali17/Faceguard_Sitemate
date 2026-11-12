"""
face_utils.py - the AI building blocks of FaceGuard
====================================================.
Following models are used;
  RetinaFace     - finds faces + 5 landmarks (eyes, nose, mouth corners)
  crop_face      - cuts a 2x-enlarged square around the face
  check_quality  - rejects photos that are too small, blurry, dark or bright
  MiniFASNet     - pretrained anti-spoofing models (real vs photo/screen)
  ArcFace        - turns a face into a 512-number "fingerprint" (embedding)
"""
from itertools import product
from math import ceil

import cv2
import numpy as np
import onnxruntime as ort

PROVIDERS = ["CPUExecutionProvider"]


# --------------------------------------------------------------------------------------
# 1. FACE DETECTION (RetinaFace, MobileNetV2 backbone)
# --------------------------------------------------------------------------------------
class RetinaFace:
    """Returns a list of faces, biggest first. Each face is a dict:
       {'box': [x1, y1, x2, y2], 'score': float, 'kps': 5x2 array of landmarks}"""

    def __init__(self, model_path, conf=0.8, max_side=640):
        self.sess = ort.InferenceSession(str(model_path), providers=PROVIDERS)
        self.conf, self.max_side = conf, max_side
        self._priors = {}

    def _anchors(self, h, w):
        # RetinaFace predicts offsets relative to fixed "anchor" boxes laid over the image
        if (h, w) not in self._priors:
            a = []
            for k, step in enumerate([8, 16, 32]):
                for i, j in product(range(ceil(h / step)), range(ceil(w / step))):
                    for size in [[16, 32], [64, 128], [256, 512]][k]:
                        a.append([(j + 0.5) * step / w, (i + 0.5) * step / h, size / w, size / h])
            self._priors[(h, w)] = np.array(a, dtype=np.float32)
        return self._priors[(h, w)]

    def detect(self, img_bgr):
        h0, w0 = img_bgr.shape[:2]
        s = min(1.0, self.max_side / max(h0, w0))                 # shrink big photos for speed
        img = cv2.resize(img_bgr, None, fx=s, fy=s) if s < 1 else img_bgr
        h, w = img.shape[:2]
        x = (img.astype(np.float32) - np.array([104, 117, 123], np.float32)).transpose(2, 0, 1)[None]
        loc, conf, landm = self.sess.run(None, {self.sess.get_inputs()[0].name: x})

        scores = conf[0][:, 1]
        keep = scores > self.conf
        if not keep.any():
            return []
        pri, loc, landm, scores = self._anchors(h, w)[keep], loc[0][keep], landm[0][keep], scores[keep]
        centre = pri[:, :2] + loc[:, :2] * 0.1 * pri[:, 2:]          # decode boxes
        size = pri[:, 2:] * np.exp(loc[:, 2:] * 0.2)
        boxes = np.hstack([centre - size / 2, centre + size / 2]) * np.array([w, h, w, h]) / s
        kps = (pri[:, None, :2] + landm.reshape(-1, 5, 2) * 0.1 * pri[:, None, 2:]) * np.array([w, h]) / s

        # Non-maximum suppression: keep one box per face
        idx = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
                               scores.tolist(), self.conf, 0.4)
        faces = [{"box": boxes[i], "score": float(scores[i]), "kps": kps[i]} for i in np.array(idx).reshape(-1)]
        return sorted(faces, key=lambda f: -(f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))


# --------------------------------------------------------------------------------------
# 2. FACE CROP (same function for training and for the app)
# --------------------------------------------------------------------------------------
def crop_face(img, box, scale=2.0, out_size=224):
    """Cut a region `scale` times bigger than the face box, kept inside the image.
    The extra border shows phone edges, paper borders and screen glare - strong spoof clues."""
    src_h, src_w = img.shape[:2]
    x, y, bw, bh = box[0], box[1], box[2] - box[0], box[3] - box[1]
    scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
    nw, nh = bw * scale, bh * scale
    cx, cy = x + bw / 2, y + bh / 2
    x1, y1, x2, y2 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > src_w - 1:
        x1 -= x2 - src_w + 1; x2 = src_w - 1
    if y2 > src_h - 1:
        y1 -= y2 - src_h + 1; y2 = src_h - 1
    return cv2.resize(img[int(y1):int(y2) + 1, int(x1):int(x2) + 1], (out_size, out_size))


# --------------------------------------------------------------------------------------
# 3. QUALITY CHECK
# --------------------------------------------------------------------------------------
def check_quality(img, face, min_size=90, min_sharpness=40, brightness=(50, 210)):
    """Return (ok, message). Bad photos make every model unreliable, so ask for a retake."""
    x1, y1, x2, y2 = [int(v) for v in face["box"]]
    h, w = img.shape[:2]
    if min(x2 - x1, y2 - y1) < min_size:
        return False, "Move closer to the camera"
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False, "Keep your whole face inside the picture"
    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < min_sharpness:
        return False, "Photo is blurry , please hold still"
    if not brightness[0] <= gray.mean() <= brightness[1]:
        return False, "Improve the lighting"
    return True, "ok"


# --------------------------------------------------------------------------------------
# 4. PRETRAINED ANTI-SPOOFING (Silent-Face MiniFASNetV2 + MiniFASNetV1SE)
# --------------------------------------------------------------------------------------
class MiniFASNet:
    """Two small CNNs looking at different crop sizes (2.7x and 4.0x of the face).
    Output classes: 0 = print attack, 1 = REAL, 2 = screen/replay attack."""

    def __init__(self, v2_path, v1se_path):
        self.models = [(ort.InferenceSession(str(v2_path), providers=PROVIDERS), 2.7),
                       (ort.InferenceSession(str(v1se_path), providers=PROVIDERS), 4.0)]

    def real_probability(self, img_bgr, face):
        probs = np.zeros(3)
        for sess, scale in self.models:
            crop = crop_face(img_bgr, face["box"], scale, 80).astype(np.float32)   # BGR, 0-255
            out = sess.run(None, {sess.get_inputs()[0].name: crop.transpose(2, 0, 1)[None]})[0][0]
            if out.min() < 0 or abs(out.sum() - 1) > 1e-3:                         # logits -> softmax
                out = np.exp(out - out.max()); out /= out.sum()
            probs += out
        return float(probs[1] / len(self.models))


# --------------------------------------------------------------------------------------
# 5. FACE RECOGNITION (ArcFace)
# --------------------------------------------------------------------------------------
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class ArcFace:
    """Aligns the face with its 5 landmarks, then returns a unit-length 512-d embedding.
    Two photos of the same person -> cosine similarity close to 1."""

    def __init__(self, model_path):
        self.sess = ort.InferenceSession(str(model_path), providers=PROVIDERS)

    def embedding(self, img_bgr, face):
        M, _ = cv2.estimateAffinePartial2D(face["kps"].astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
        aligned = cv2.warpAffine(img_bgr, M, (112, 112))
        x = (aligned[:, :, ::-1].astype(np.float32) - 127.5) / 127.5
        e = self.sess.run(None, {self.sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)
