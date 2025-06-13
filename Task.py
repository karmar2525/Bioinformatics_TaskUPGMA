"""
This program calculates a phylogenetic tree using the UPGMA method.
It supports two types of input data:
  1) A set of DNA sequences (from a FASTA file or directly via command line)
  2) A pre-computed distance matrix (CSV format)

Functionality:
- Validates DNA sequences for correctness.
- Computes pairwise global alignment distances using customizable scoring parameters:
  match score, mismatch penalty, and gap penalty.
- Constructs a distance matrix if sequences are given.
- Performs UPGMA clustering to generate a phylogenetic tree.
- Saves the tree as a graphical dendrogram image.
- Generates a PDF report summarizing input sequences, distance matrix,
  linkage matrix, and the phylogenetic tree visualization.

Usage example:
  python upgma.py --file sequences.fasta --match 1 --mismatch -1 --gap -2 --out report.pdf
  python upgma.py --matrix distance_matrix.csv --out report.pdf

Command line arguments:
  --file  - Input FASTA file with DNA sequences.
  --seqs -  List of DNA sequences provided inline.
  --matrix -  CSV file with pre-computed distance matrix.
  --match  - Match score for alignment (default: 1).
  --mismatch - Mismatch penalty for alignment (default: -1).
  --gap - Gap penalty for alignment (default: -1).
  --out  - Output PDF report filename (default: upgma_report.pdf).
"""

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

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Dummy:
        def __getattr__(self, _): return ''
    Fore = Style = Dummy()


def validate_dna(seq):
    """
    Check if a DNA sequence contains only valid nucleotide characters (A, C, G, T).

    Args:
        seq (str): DNA sequence.

    Returns:
        bool: True if valid, False otherwise.
    """
    return all(base in 'ACGTacgt' for base in seq)


def load_sequences(file=None, seqs=None):
    """
    Load DNA sequences from a FASTA file or from a provided list.

    Args:
        file (str or None): Path to a FASTA file.
        seqs (list of str or None): List of DNA sequences.

    Returns:
        tuple: (list of sequences, list of sequence labels)

    Raises:
        ValueError: If less than two valid sequences are provided.
    """
    sequences, labels = [], []
    if file:
        for record in SeqIO.parse(file, "fasta"):
            seq = str(record.seq).upper()
            if validate_dna(seq):
                sequences.append(seq)
                labels.append(record.id)
            else:
                print(f"{Fore.YELLOW}Warning: Invalid sequence skipped: {record.id}")
    elif seqs:
        for i, s in enumerate(seqs):
            if validate_dna(s):
                sequences.append(s.upper())
                labels.append(f"Seq{i+1}")
            else:
                print(f"{Fore.YELLOW}Warning: Invalid sequence skipped: {s}")
    else:
        raise ValueError("Provide input using --file or --seqs.")

    if len(sequences) < 2:
        raise ValueError("At least two valid DNA sequences required.")
    return sequences, labels


def compute_distance(seq1, seq2, match=1, mismatch=-1, gap=-1):
    """
    Compute the distance between two DNA sequences using global pairwise alignment.

    Args:
        seq1 (str): First DNA sequence.
        seq2 (str): Second DNA sequence.
        match (int): Score for a match.
        mismatch (int): Penalty for a mismatch.
        gap (int): Penalty for a gap (both opening and extension).

    Returns:
        float: Distance value between 0 and 1 (1 - identity).
    """
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


def build_distance_matrix(sequences, match=1, mismatch=-1, gap=-1):
    """
    Build a symmetric distance matrix from a list of sequences.

    Args:
        sequences (list of str): DNA sequences.
        match (int): Match score.
        mismatch (int): Mismatch penalty.
        gap (int): Gap penalty.

    Returns:
        numpy.ndarray: 2D array with pairwise distances.
    """
    n = len(sequences)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_distance(sequences[i], sequences[j], match, mismatch, gap)
            matrix[i][j] = matrix[j][i] = dist
    return matrix


def upgma_clustering(dist_matrix, labels, img_path="upgma_tree.png"):
    """
    Perform UPGMA hierarchical clustering and save dendrogram plot.

    Args:
        dist_matrix (numpy.ndarray): Distance matrix.
        labels (list of str): Labels for sequences.
        img_path (str): Output file path for dendrogram image.

    Returns:
        numpy.ndarray: Linkage matrix from clustering.
    """
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
    """
    Add a styled section header to the PDF.

    Args:
        pdf (FPDF): PDF object.
        title (str): Title of the section.
    """
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_draw_color(0, 0, 128)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    pdf.set_text_color(0, 0, 0)


def check_page_space(pdf, line_height=6, margin=10):
    """
    Check if there is enough space left on the current PDF page, else add new page.

    Args:
        pdf (FPDF): PDF object.
        line_height (int): Height of one line.
        margin (int): Bottom margin.
    """
    if pdf.get_y() > (pdf.h - margin - line_height):
        pdf.add_page()


def generate_pdf(sequences, labels, dist_matrix, linkage_matrix, tree_img_path, output_pdf="upgma_report.pdf", match=1, mismatch=-1, gap=-1):
    """
    Generate a PDF report containing input sequences, distance matrix,
    linkage matrix, the phylogenetic tree image, and alignment parameters.

    Args:
        sequences (list of str): Input DNA sequences.
        labels (list of str): Labels for sequences.
        dist_matrix (numpy.ndarray): Distance matrix.
        linkage_matrix (numpy.ndarray): Linkage matrix from clustering.
        tree_img_path (str): Path to dendrogram image.
        output_pdf (str): Output PDF filename.
        match (int): Match score used in alignment.
        mismatch (int): Mismatch penalty used in alignment.
        gap (int): Gap penalty used in alignment.
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 150)
    pdf.cell(0, 15, "UPGMA Phylogenetic Tree Report", align='C', ln=True)
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)

    # Dodanie sekcji z parametrami dopasowania
    add_section_header(pdf, "Alignment Parameters")
    pdf.set_font("Courier", size=10)
    pdf.cell(0, 6, f"Match score: {match}", ln=True)
    pdf.cell(0, 6, f"Mismatch penalty: {mismatch}", ln=True)
    pdf.cell(0, 6, f"Gap penalty: {gap}", ln=True)
    pdf.ln(5)

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

        if pdf.get_y() + img_height_mm + 14 > pdf.h - 10:
            pdf.add_page()

        add_section_header(pdf, "Phylogenetic Tree (Dendrogram)")
        pdf.image(tree_img_path, x=10, w=img_width_mm)
    else:
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 10, "Tree image file not found.", ln=True)

    pdf.output(output_pdf)
    print(f"Report saved to {output_pdf}")



def load_distance_matrix_from_csv(file):
    """
    Load a distance matrix from a CSV file.

    Args:
        file (str): CSV filename.

    Returns:
        tuple: (distance matrix (numpy.ndarray), list of labels)
    """
    df = pd.read_csv(file, index_col=0)
    labels = list(df.index)
    matrix = df.values
    return matrix, labels


def main():
    parser = argparse.ArgumentParser(description="UPGMA Phylogenetic Tree Builder")
    parser.add_argument("--file", type=str, help="Input FASTA file with DNA sequences")
    parser.add_argument("--seqs", nargs='+', help="DNA sequences inline")
    parser.add_argument("--matrix", type=str, help="CSV file with pre-computed distance matrix")
    parser.add_argument("--match", type=int, default=1, help="Match score for alignment (default=1)")
    parser.add_argument("--mismatch", type=int, default=-1, help="Mismatch penalty for alignment (default=-1)")
    parser.add_argument("--gap", type=int, default=-1, help="Gap penalty for alignment (default=-1)")
    parser.add_argument("--out", type=str, default="upgma_report.pdf", help="Output PDF report filename")

    args = parser.parse_args()

    if args.matrix:
        dist_matrix, labels = load_distance_matrix_from_csv(args.matrix)
        sequences = None
    else:
        sequences, labels = load_sequences(args.file, args.seqs)
        dist_matrix = build_distance_matrix(sequences, args.match, args.mismatch, args.gap)

    tree_img_path = "upgma_tree.png"
    linkage_matrix = upgma_clustering(dist_matrix, labels, tree_img_path)
    generate_pdf(
        sequences if sequences else [],
        labels,
        dist_matrix,
        linkage_matrix,
        tree_img_path,
        args.out,
        match=args.match,
        mismatch=args.mismatch,
        gap=args.gap
    )


if __name__ == "__main__":
    main()
