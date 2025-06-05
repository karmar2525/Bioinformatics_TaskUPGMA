import argparse
import os
from datetime import datetime
from io import BytesIO
from Bio.Align import PairwiseAligner
from Bio import SeqIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from fpdf import FPDF
from PIL import Image


def validate_dna(seq):
    return all(base in 'ACGTacgt' for base in seq)


def load_sequences(file=None, seqs=None):
    sequences, labels = [], []
    if file:
        for record in SeqIO.parse(file, "fasta"):
            seq = str(record.seq).upper()
            if validate_dna(seq):
                sequences.append(seq)
                labels.append(record.id)
            else:
                print(f"Warning: Invalid sequence skipped: {record.id}")
    elif seqs:
        for i, s in enumerate(seqs):
            if validate_dna(s):
                sequences.append(s.upper())
                labels.append(f"Seq{i+1}")
            else:
                print(f"Warning: Invalid sequence skipped: {s}")
    else:
        raise ValueError("Provide input using --file or --seqs.")

    if len(sequences) < 2:
        raise ValueError("At least two valid DNA sequences required.")

    return sequences, labels


def compute_distance(seq1, seq2, match=1, mismatch=-1, gap=-1):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = gap
    aligner.extend_gap_score = gap
    score = aligner.score(seq1, seq2)
    max_score = min(len(seq1), len(seq2)) * match
    identity = score / max_score if max_score else 0
    return 1 - identity


def build_distance_matrix(sequences, match=1, mismatch=0, gap=-1):
    n = len(sequences)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_distance(sequences[i], sequences[j], match, mismatch, gap)
            matrix[i][j] = matrix[j][i] = dist
    return matrix


def upgma_clustering(dist_matrix, labels, img_path="upgma_tree.png"):
    linkage_matrix = linkage(squareform(dist_matrix), method='average')
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=labels, orientation='top')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title("UPGMA Phylogenetic Tree")
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()
    return linkage_matrix


def add_section_header(pdf, title):
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_draw_color(0, 0, 128)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    pdf.set_text_color(0, 0, 0)


def check_page_space(pdf, line_height=6, margin=10):
    if pdf.get_y() > (pdf.h - margin - line_height):
        pdf.add_page()


def generate_pdf(sequences, labels, dist_matrix, linkage_matrix, tree_img_path, output_pdf="upgma_report.pdf"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 150)
    pdf.cell(0, 15, "UPGMA Phylogenetic Tree Report", align='C', ln=True)
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)

    add_section_header(pdf, "Input Sequences")
    pdf.set_font("Courier", size=9)
    for label, seq in zip(labels, sequences):
        pdf.multi_cell(0, 6, f"{label}: {seq}")
    pdf.ln(5)

    add_section_header(pdf, "Distance Matrix")
    pdf.set_font("Courier", size=8)
    header = "".ljust(10) + "".join([f"{lbl[:6]:>8}" for lbl in labels])
    pdf.cell(0, 6, header, ln=True)
    for lbl, row in zip(labels, dist_matrix):
        row_str = "".join([f"{val:8.3f}" for val in row])
        pdf.cell(0, 6, f"{lbl[:6]:<8}{row_str}", ln=True)
    pdf.ln(5)

    add_section_header(pdf, "Linkage Matrix")
    pdf.set_font("Courier", size=8)
    for row in linkage_matrix:
        row_str = "  ".join([f"{v:7.3f}" for v in row])
        pdf.cell(0, 6, row_str, ln=True)
    pdf.ln(5)

    if os.path.isfile(tree_img_path):
        with Image.open(tree_img_path) as img:
            width_px, height_px = img.size
        img_width_mm = 180
        scale = img_width_mm / width_px
        img_height_mm = height_px * scale

        header_height = 14
        total_height = header_height + img_height_mm

        if pdf.get_y() + total_height > pdf.h - 10:
            pdf.add_page()

        add_section_header(pdf, "Phylogenetic Tree (Dendrogram)")
        pdf.image(tree_img_path, x=10, y=pdf.get_y(), w=img_width_mm)
    else:
        add_section_header(pdf, "Phylogenetic Tree (Dendrogram)")
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, "Error: Dendrogram image not found.", ln=True)

    pdf.output(output_pdf)
    print(f"PDF report saved as {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description="UPGMA Phylogenetic Tree Builder")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", help="FASTA file with DNA sequences")
    input_group.add_argument("--seqs", nargs='+', help="List of sequences")
    input_group.add_argument("--matrix", help="CSV with distance matrix")

    parser.add_argument("--match", type=int, default=1, help="Match score")
    parser.add_argument("--mismatch", type=int, default=-1, help="Mismatch penalty")
    parser.add_argument("--gap", type=int, default=-1, help="Gap penalty")

    args = parser.parse_args()

    if args.matrix:
        df = pd.read_csv(args.matrix, index_col=0)
        dist_matrix = df.values
        labels = list(df.index)
        sequences = ["" for _ in labels]
        print("Distance matrix loaded from CSV.")
    else:
        sequences, labels = load_sequences(file=args.file, seqs=args.seqs)
        dist_matrix = build_distance_matrix(sequences, args.match, args.mismatch, args.gap)
        pd.DataFrame(dist_matrix, index=labels, columns=labels).to_csv("distance_matrix.csv")
        print("Pairwise distances computed and saved to distance_matrix.csv")

    tree_img = "upgma_tree.png"
    report_pdf = "upgma_report.pdf"

    linkage_matrix = upgma_clustering(dist_matrix, labels, img_path=tree_img)
    generate_pdf(sequences, labels, dist_matrix, linkage_matrix, tree_img, output_pdf=report_pdf)

    print(f"Analysis complete. Outputs:\n- {tree_img}\n- {report_pdf}")


if __name__ == "__main__":
    main()