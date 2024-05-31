# Python 3.11.5.
# FILE: texto.py
# AUTHOR: José Luis López Carmona
# CREATE DATE: 28/01/2024
from Bio import SeqIO
import sys

def extract_full_taxonomy_info(fasta_file_path, output_file_path):
    with open(output_file_path, 'w') as output_file:
        output_file.write("ID, Phylum, Order, Family, Genus\n")
        
        for record in SeqIO.parse(fasta_file_path, "fasta"):
            header_parts = record.description.split()  
            if len(header_parts) >= 5:    # Example:
                phylum = header_parts[1]  # Alphaproteobacteria
                order = header_parts[2]   # Rhodobacterales
                family = header_parts[3]  # Rhodobacteraceae
                genus = header_parts[4]   # Rhodobaca
                output_file.write(f"{record.id}, {phylum}, {order}, {family}, {genus}\n")
            else:
                print(f"Warning: Missing data for record {record.id}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python texto.py <input_fasta_file> <output_file>")
        sys.exit(1)

    input_fasta_file = sys.argv[1]
    output_text_file = sys.argv[2]
    extract_full_taxonomy_info(input_fasta_file, output_text_file)

# Use this code into bash:    
# python texto.py data\Milseq.fasta micro.txt
