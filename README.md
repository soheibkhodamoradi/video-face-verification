# Video Face Verification

A CPU-only face verification project using **RetinaFace** detection and **ArcFace** embeddings from InsightFace's pretrained `buffalo_l` model pack. Given two videos (or two still images during evaluation), it estimates whether they show the same person.

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

Image pairs (including LFW-style pairs) use the same detector, quality filter, alignment, and embedding code, but no frame sampling. Video pairs use the selected aggregation strategy.

The script prints and saves:

- **Accuracy** at the supplied threshold.
- **ROC curve** and **AUC** (ranking quality over all thresholds).
- **EER** and the threshold nearest the equal-error point.
- **FAR**: different-person pairs incorrectly accepted, divided by all different-person pairs.
- **FRR**: same-person pairs incorrectly rejected, divided by all same-person pairs.

`outputs/roc_curve.png`, `outputs/metrics.json`, and `outputs/scores.csv` are generated. Use the recommended EER threshold as a starting point; for a real application, confirm it on a separate validation/test split and choose an operating point appropriate for the acceptable FAR/FRR trade-off.

## Google Colab

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

## Results

No accuracy number is claimed here because this repository intentionally contains no personal videos or face datasets. Performance depends strongly on the evaluation data and chosen threshold. Run `evaluate.py` on your labeled pairs and report its accuracy, AUC, EER, FAR, and FRR alongside the dataset description.

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
