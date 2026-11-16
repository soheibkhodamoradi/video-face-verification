# Video Face Verification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soheibkhodamoradi/video-face-verification/blob/main/notebooks/colab_demo.ipynb)

A CPU-only face verification project using **RetinaFace/SCRFD** detection and **ArcFace** embeddings from InsightFace's pretrained `buffalo_l` model pack. Given two videos (or two still images during evaluation), it estimates whether they show the same person.

No model is trained in this project. The pretrained ONNX models run with `onnxruntime` and `CPUExecutionProvider`.

## Pipeline

1. Sample every Nth video frame (frame 0 is included; N defaults to 5).
2. Detect faces with RetinaFace and select only the largest face.
3. Reject frames with no face, a box smaller than 80 px, detector confidence below 0.5, or variance of Laplacian below 50.
4. Align the face using its five landmarks and compute a 512-dimensional ArcFace embedding.
5. L2-normalize each embedding and compare the two videos using either:
   - **`mean`**: average each video's normalized embeddings, re-normalize each average, then compute cosine similarity.
   - **`pairwise`**: average the cosine similarities of every frame embedding in video 1 with every frame embedding in video 2.
6. Report **same person** when similarity is greater than or equal to the threshold, otherwise **different person**.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The first run downloads `buffalo_l` automatically to `~/.insightface/models/`. An internet connection is needed for that first run; subsequent runs reuse the local models. All inference is on CPU.

## Verify two videos

```bash
python -m face_verify.cli video1.mp4 video2.mp4 \
  --strategy mean --threshold 0.4 --every 5
```

Example output:

```text
Similarity: 0.512345
Decision: same person (threshold 0.400)
Usable frames: video1=18, video2=16
```

Use `--strategy pairwise` for the pairwise strategy. Quality controls are also available:

```bash
python -m face_verify.cli video1.mp4 video2.mp4 \
  --min-face-size 80 --min-detection-score 0.5 --min-blur-score 50
```

Run `python -m face_verify.cli --help` for all options. Missing files, unreadable videos, and inputs with no usable face produce explicit error messages.

> **Important:** 0.4 is only a starting value, not a universal threshold. Choose a threshold using labeled data from your own camera, population, and operating conditions.

### Phone videos

Phone and front-camera videos are often soft or heavily compressed even when faces are large and frontal, so the default blur threshold (`--min-blur-score 50`) may reject every frame with `No usable face found`. Lower `--min-blur-score` (for example to `10`) when working with phone footage:

```bash
python -m face_verify.cli video1.mp4 video2.mp4 \
  --strategy mean --threshold 0.4 --every 5 --min-blur-score 10
```

## Evaluation and threshold selection

Create a CSV with the exact header `path1,path2,label`. Labels `1` and `same` mean the same person; `0` and `different` mean different people. Relative paths are resolved relative to the CSV file.

```csv
path1,path2,label
videos/alice_1.mp4,videos/alice_2.mp4,1
videos/alice_1.mp4,videos/bob_1.mp4,0
lfw/Aaron_Peirsol_0001.jpg,lfw/Aaron_Peirsol_0002.jpg,1
lfw/Aaron_Peirsol_0001.jpg,lfw/Adam_Sandler_0001.jpg,0
```

Run:

```bash
python evaluate.py pairs.csv --strategy mean --threshold 0.4 --every 5 \
  --output-dir outputs
```

`evaluate.py` also accepts the same quality filters as the CLI (`--min-face-size`, `--min-detection-score`, `--min-blur-score`), for example `--min-blur-score 10` on phone videos.

Image pairs (including LFW-style pairs) use the same detector, quality filter, alignment, and embedding code, but no frame sampling. Video pairs use the selected aggregation strategy.

The script prints and saves:

- **Accuracy** at the supplied threshold.
- **ROC curve** and **AUC** (ranking quality over all thresholds).
- **EER** and the threshold nearest the equal-error point.
- **FAR**: different-person pairs incorrectly accepted, divided by all different-person pairs.
- **FRR**: same-person pairs incorrectly rejected, divided by all same-person pairs.

`outputs/roc_curve.png`, `outputs/metrics.json`, and `outputs/scores.csv` are generated. Use the recommended EER threshold as a starting point; for a real application, confirm it on a separate validation/test split and choose an operating point appropriate for the acceptable FAR/FRR trade-off.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soheibkhodamoradi/video-face-verification/blob/main/notebooks/colab_demo.ipynb)

Open [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb) in Colab. The notebook installs the project, prompts for two video uploads, runs verification, displays the score and decision, and shows a detected face crop from each video. Do not save personal videos or face images into the repository.

## Optional Gradio app

```bash
python app.py
```

The app accepts two video uploads and exposes the strategy, threshold, and frame-sampling interval. It listens on `0.0.0.0`, so it also works in remote notebook/container environments.

## Tests

The tests use only small synthetic arrays and do not download InsightFace models:

```bash
pytest
```

They cover cosine similarity, normalization, aggregation strategies, threshold decisions, blur scoring, and input validation.

## Preliminary results

> **Note:** The numbers below come from a tiny local sanity check on **two video pairs** (one same-person pair and one different-person pair). This is **not** a formal evaluation, and **no accuracy, ROC, AUC, or EER is claimed** or has been computed yet. No personal videos, names, photos, or video frames are included in this repository.

Local CPU-only run on phone front-camera videos (~3 s, 720×1280):

```bash
python -m face_verify.cli A.mp4 B.mp4 --strategy mean --threshold 0.4 --every 5 --min-blur-score 10
```

- **Same-person pair (two videos of the same person):** cosine similarity `~0.818`, decision `"same person"`, usable frames `21` and `17` (with `--every 2` and the same relaxed filters: similarity `~0.819`, `52` and `42` usable frames).
- **Different-person pair (one video each of two different people):** cosine similarity `~-0.017`, decision `"different person"`, usable frames `21` and `14`.
- **Default quality filters (`--min-blur-score 50`):** rejected every frame (`No usable face found`) despite large, frontal faces because phone videos are soft/compressed; setting `--min-blur-score 10` fixed it.

Performance depends strongly on the evaluation data and chosen threshold. Collect a representative labeled dataset and run `evaluate.py` before drawing conclusions about accuracy, AUC, EER, FAR, or FRR.

## Responsible use

- **Biometric data:** Facial images, video frames, and face embeddings are sensitive biometric identifiers. Handle, store, and delete them responsibly, and never commit personal videos or face images to Git.
- **Informed consent:** Always obtain explicit, informed consent from every person recorded before capturing or processing their video, in accordance with applicable privacy and biometric-data regulations.
- **Not for high-stakes decisions:** This repository is a lightweight research/educational baseline and must **not** be used for high-stakes or automated decisions, including security or access control, law enforcement or legal proceedings, surveillance, or hiring and employment screening.
- **Demographic and condition disparities:** Face detection and recognition models can show different false-acceptance and false-rejection rates across demographic groups (such as skin tone, age, and gender presentation) and across recording conditions (camera quality, compression, lighting, pose, and occlusion).

## Limitations

- Only the largest detected face is used; other people in the frame are ignored. This is not a multi-person tracking system.
- Small, blurry, occluded, profile, or poorly lit faces may be rejected or produce unreliable embeddings.
- The blur threshold and face-size threshold may need adjustment for different resolutions and cameras.
- Sampling can miss the best frames; decreasing `--every` improves coverage but increases CPU time. Pairwise comparison costs `O(n*m)` embedding comparisons for `n` and `m` usable frames.
- Face verification can reflect dataset and demographic biases. Evaluate on representative data and do not use the default threshold for high-stakes decisions.
- Still-image evaluation measures image-pair performance; it does not replace validation on the target videos.
- Videos, face images, generated outputs, and model weights are intentionally excluded from Git.

## Project layout

```text
src/face_verify/
  frames.py       # frame sampling and image reading
  detector.py     # RetinaFace detection and quality filtering
  embedder.py     # landmark alignment and ArcFace inference
  aggregate.py    # normalization, mean/pairwise aggregation, cosine similarity
  verify.py       # complete pipeline and result types
  cli.py          # command-line interface
evaluate.py       # ROC/AUC/EER/FAR/FRR evaluation
app.py            # optional Gradio UI
notebooks/        # Google Colab demo
tests/            # lightweight unit tests
```

## References

- **InsightFace:** Jia Guo, Jiankang Deng, et al., *InsightFace: 2D and 3D Face Analysis Project*, <https://github.com/deepinsight/insightface>.
- **ArcFace:** Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," *CVPR*, 2019.
- **RetinaFace:** Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou, "RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild," *CVPR*, 2020.
- **SCRFD:** Jia Guo, Jiankang Deng, Alexandros Lattas, and Stefanos Zafeiriou, "Sample and Computation Redistribution for Efficient Face Detection," *ICLR*, 2022.

## License

The source code in this repository is released under the MIT License (see `LICENSE`).

This project uses the pretrained InsightFace `buffalo_l` models, which are provided for
non-commercial research purposes only. The models are downloaded at runtime and are not
included in this repository. Check InsightFace's license terms before any commercial use.
