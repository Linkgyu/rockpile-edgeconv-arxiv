from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "arxiv_manuscript.docx"
FIG = ROOT / "manuscript" / "figures"


TITLE = "A Synthetic Exterior Point-Cloud Benchmark for Rockpile Fragmentation and P80 Estimation with EdgeConv Edge Affinity"
AUTHOR = "Namgyu (Noah) Ha"
AFFILIATION = "Department of Energy and Resources Engineering, National Korea Maritime and Ocean University, Busan, Republic of Korea"


ABSTRACT = (
    "Operational fragmentation monitoring increasingly uses photogrammetric or lidar point clouds, but labelled real "
    "rockpile scans with point-level fragment identity and independent particle-size-distribution (PSD) ground truth "
    "remain scarce. This study presents a reproducible synthetic exterior point-cloud benchmark for rockpile surface "
    "clustering and P80 estimation. One hundred no-boundary synthetic rockpile scenes were generated with 150 requested "
    "fragments per scene using a physics-informed exterior-surface construction workflow, then split at the scene level "
    "into 60 training, 20 validation, and 20 held-out test scenes. A DGCNN-style EdgeConv edge-affinity model was trained "
    "to predict whether neighbouring exterior points belong to the same fragment. Connected components of high-affinity "
    "edges were converted into surface-cluster PSD proxies, with the edge threshold selected on validation scenes by P80 "
    "error and noise-aware post-processing. The model learned edge affinity reliably, reaching validation average "
    "precision of 0.9313 at epoch 24. However, downstream PSD estimation remained sensitive to threshold calibration: "
    "validation selected the post-split EdgeConv variant at threshold 0.997, and the held-out test mean absolute P80 "
    "error was 19.04% with a mean noise fraction of 0.603. These results indicate that EdgeConv provides a useful "
    "controlled benchmark for surface-cluster affinity learning, while robust post-processing and field calibration "
    "remain the main barriers to operational P80 estimation."
)


SECTIONS = [
    (
        "Introduction",
        [
            "Post-blast fragmentation affects loading, hauling, crushing, and comminution performance. Classical blast-fragmentation models such as Kuz-Ram and Swebrec remain useful because particle size distribution links blasting outcomes with mine-to-mill performance.",
            "Image-based fragmentation systems are widely used, but two-dimensional images must infer scale, occlusion, and particle overlap from a projected surface. Three-dimensional photogrammetry and lidar workflows reduce some of these limitations, but field point clouds rarely include reliable point-level fragment labels.",
            "This paper reports a compact benchmark built around a controlled synthetic stage. The goal is not to claim field-ready fragmentation measurement. Instead, the benchmark isolates a narrower question: given labelled exterior synthetic rockpile scans, how well can a DGCNN/EdgeConv edge-affinity model learn surface grouping signals, and how does that learning translate into P80 estimates after connected-component post-processing?",
        ],
    ),
    (
        "Synthetic Exterior Rockpile Dataset",
        [
            "The reported dataset contains 100 synthetic rockpile scenes. Each scene requested 150 fragments and retained only the exterior point-cloud surface used for learning. The final scene index contains an average of 146.68 visible fragments, 5735.63 exterior points, a mean base radius of 0.898 m, and a mean pile height of 1.084 m. Scenes were split by scene identifier, not by point, into 60 training scenes, 20 validation scenes, and 20 held-out test scenes.",
            "The final production dataset used a no-boundary envelope-relax configuration rather than a full sequential-drop Chrono simulation. Full dynamic sequential-drop variants were tested with convex-hull and clump contact, but they either required impractical computation for 100 scenes or produced numerical outliers when boundary walls were removed.",
        ],
    ),
    (
        "Edge-Affinity Model",
        [
            "Fragment identity is scene-specific, so the learning problem is formulated as edge affinity. For each local graph edge between neighbouring exterior points, the model predicts whether the two endpoints belong to the same ground-truth fragment.",
            "The model follows the Dynamic Graph CNN/EdgeConv principle of learning local point-cloud features from graph neighbourhoods. Point features include normalized coordinates, normals, and curvature; edge attributes include geometric differences and local surface cues. During training, a balanced subset of positive and negative edges is sampled from each scene. Photogrammetry-realism augmentation perturbs point positions, normals, curvature, and edge geometry while preserving graph labels.",
        ],
    ),
    (
        "PSD Proxy and Post-Processing",
        [
            "At inference time, predicted edge probabilities are thresholded. Retained edges define connected components, and each component is interpreted as a visible surface cluster. The cluster is not assumed to be a clean full fragment instance. Instead, it is converted into a PSD proxy using a surface-cluster diameter estimate and an equivalent proxy volume.",
            "Four prediction variants are evaluated: raw EdgeConv components, absorbed components where unlabelled points are reassigned by edge affinity, height-marker post-splitting of oversized clusters, and the combined absorb-plus-post-split variant. Thresholds are swept on the 20 validation scenes.",
        ],
    ),
    (
        "Training and Validation Results",
        [
            "The EdgeConv model was trained for 24 epochs with 60 training scenes. Validation metrics were computed on 12 validation scenes per epoch for efficiency. Training loss decreased from 0.5722 to 0.3479, while validation average precision increased from 0.8569 to 0.9313. Validation ROC-AUC reached 0.9179 at epoch 24.",
            "Validation threshold selection showed the main failure mode. Lower thresholds improve component connectivity but merge neighbouring fragments; higher thresholds reduce merging but split many points into small or unlabelled components. The selected operating point was the post-split EdgeConv variant at threshold 0.997.",
        ],
    ),
    (
        "Held-Out Test Results",
        [
            "The selected validation setting was frozen and evaluated on the 20 held-out test scenes. The selected post-split EdgeConv variant achieved a mean absolute P80 error of 19.04% and a median absolute P80 error of 18.36%. The raw EdgeConv variant gave 20.82% mean absolute P80 error. Absorption reduced noise fraction from about 0.60 to about 0.51 but worsened P80 error in this run.",
        ],
    ),
    (
        "Discussion",
        [
            "The results are clear but modest. The network learns edge affinity, but downstream PSD estimation is weaker because thresholded connected components are brittle. The validation procedure selected a very high edge threshold, 0.997, and the held-out noise fraction remained approximately 60%.",
            "The no-boundary 150-fragment dataset is more stable than the attempted sequential-drop Chrono scenes and is practical for 100-scene training. Nevertheless, it is still synthetic and still uses a surface-cluster PSD proxy. Real muckpile deployment requires independent field PSD references and a frozen, pre-specified calibration protocol.",
        ],
    ),
    (
        "Reproducibility",
        [
            "The repository accompanying this manuscript includes the scene index, summary CSV files, training script, evaluation script, figures, and manuscript source. The full generated .npz scenes and model checkpoint are treated as regenerated artifacts rather than committed source files.",
            "The reported run used 100 scenes, a 60/20/20 scene-level split, 24 training epochs, 22,000 maximum training edges per scene, 36,000 maximum validation edges per scene, and photogrammetry-realism augmentation strength 0.75.",
        ],
    ),
    (
        "Conclusion",
        [
            "This benchmark shows that EdgeConv edge affinity can learn useful local surface relationships in labelled synthetic rockpile exterior scans. The best validation AP reached 0.9313, but the held-out mean absolute P80 error remained 19.04% under a high-threshold post-split connected-component rule. The main research bottleneck is therefore no longer only synthetic data generation or edge-affinity learning; it is robust calibration from edge probabilities to PSD-relevant components.",
        ],
    ),
]


REFERENCES = [
    "Cunningham, C. V. B. The Kuz-Ram model for prediction of fragmentation from blasting. 1st International Symposium on Rock Fragmentation by Blasting, 1983.",
    "Ouchterlony, F. The Swebrec function: linking fragmentation by blasting and crushing. Mining Technology, 2005.",
    "Engin, I. C.; Maerz, N. H.; Boyko, K. J.; Reals, R. Practical measurement of size distribution of blasted rocks using LiDAR scan data. Rock Mechanics and Rock Engineering, 2020.",
    "Wang, Y.; Sun, Y.; Liu, Z.; Sarma, S. E.; Bronstein, M. M.; Solomon, J. M. Dynamic Graph CNN for learning on point clouds. ACM Transactions on Graphics, 2019.",
    "Qi, C. R.; Su, H.; Mo, K.; Guibas, L. J. PointNet: Deep learning on point sets for 3D classification and segmentation. CVPR, 2017.",
    "Qi, C. R.; Yi, L.; Su, H.; Guibas, L. J. PointNet++: Deep hierarchical feature learning on point sets in a metric space. NeurIPS, 2017.",
    "Ester, M.; Kriegel, H.-P.; Sander, J.; Xu, X. A density-based algorithm for discovering clusters in large spatial databases with noise. KDD, 1996.",
    "Pedregosa, F. et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 2011.",
    "Paszke, A. et al. PyTorch: An imperative style, high-performance deep learning library. NeurIPS, 2019.",
]


def set_styles(doc: Document) -> None:
    sec = doc.sections[0]
    sec.top_margin = Inches(1)
    sec.bottom_margin = Inches(1)
    sec.left_margin = Inches(1)
    sec.right_margin = Inches(1)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.18

    for name, size, color in [
        ("Heading 1", 16, RGBColor(46, 116, 181)),
        ("Heading 2", 13, RGBColor(46, 116, 181)),
    ]:
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        style.paragraph_format.space_after = Pt(6)


def add_centered(doc: Document, text: str, size: int, bold: bool = False) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold


def add_figure(doc: Document, filename: str, caption: str, width: float = 5.5) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG / filename), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.paragraph_format.space_after = Pt(10)


def add_results_table(doc: Document) -> None:
    rows = [
        ("EdgeConv post split", "0.997", "19.04", "18.36", "0.603"),
        ("EdgeConv raw", "0.997", "20.82", "19.42", "0.603"),
        ("EdgeConv absorb + post split", "0.997", "21.84", "21.40", "0.506"),
        ("EdgeConv absorb", "0.997", "23.45", "21.40", "0.506"),
    ]
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    headers = ["Variant", "Threshold", "Mean abs. P80 error (%)", "Median abs. P80 error (%)", "Noise fraction"]
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = text
        for par in cell.paragraphs:
            for run in par.runs:
                run.bold = True
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = text
    cap = doc.add_paragraph("Table 1. Held-out test summary for the 20 test scenes.")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True


def build() -> None:
    doc = Document()
    set_styles(doc)
    add_centered(doc, TITLE, 17, bold=True)
    add_centered(doc, AUTHOR, 11)
    add_centered(doc, AFFILIATION, 10)

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(ABSTRACT)
    doc.add_paragraph(
        "Keywords: rock fragmentation; point cloud; synthetic benchmark; EdgeConv; DGCNN; particle size distribution; P80; photogrammetry; mining"
    )

    for title, paragraphs in SECTIONS:
        doc.add_heading(title, level=1)
        for text in paragraphs:
            doc.add_paragraph(text)
        if title == "Synthetic Exterior Rockpile Dataset":
            add_figure(
                doc,
                "dem_noboundary_relax150_scene000_preview.png",
                "Figure 1. Representative exterior point cloud from the 150-fragment no-boundary synthetic dataset.",
                width=5.7,
            )
        elif title == "Training and Validation Results":
            add_figure(
                doc,
                "02_edgeconv_training_curve.png",
                "Figure 2. Training loss and validation average precision for the 24-epoch EdgeConv run.",
                width=5.5,
            )
        elif title == "Held-Out Test Results":
            add_results_table(doc)
            add_figure(
                doc,
                "03_edgeconv_test_p80_error_histogram.png",
                "Figure 3. Held-out test distribution of absolute P80 error for the selected EdgeConv post-split setting.",
                width=5.0,
            )

    doc.add_heading("Acknowledgments", level=1)
    doc.add_paragraph(
        "During preparation of this manuscript, the author used OpenAI Codex for drafting, coding, formatting, and reproducibility assistance. The author reviewed and edited the output and takes responsibility for the manuscript."
    )
    doc.add_heading("Data and Code Availability", level=1)
    doc.add_paragraph(
        "Code, manuscript source, figures, and summary tables are prepared for release at https://github.com/Linkgyu/rockpile-edgeconv-arxiv. The full generated scenes can be regenerated using the included scripts."
    )
    doc.add_heading("References", level=1)
    for i, ref in enumerate(REFERENCES, start=1):
        doc.add_paragraph(f"{i}. {ref}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
