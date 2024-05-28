# Python 3.11.5
# FILE: CNN_pro.py
# AUTHOR: José Luis López Carmona
# CREATE DATE: 27/01/2024
# Libraries
import sys
import math
import numpy as np

# Creating k-mers alphabet
def make_kmer_list (k, alphabet):

    # Base case.
    if (k == 1):
        return(alphabet)

    # Handle k=0 from user.
    if (k == 0):
        return([])

    # Error case.
    if (k < 1):
        sys.stderr.write("Invalid k=%d" % k)
        sys.exit(1)

    # Precompute alphabet length for speed.
    alphabet_length = len(alphabet)

    # Recursive call.
    return_value = []
    for kmer in make_kmer_list(k-1, alphabet):
        for i_letter in range(0, alphabet_length):
            return_value.append(kmer + alphabet[i_letter])
              
    return(return_value)

##############################################################################
def make_upto_kmer_list (k_values,
                         alphabet):

    # Compute the k-mer for each value of k.
    return_value = []
    for k in k_values:
        return_value.extend(make_kmer_list(k, alphabet))

    return(return_value)

##############################################################################
# Don't take
def make_to_kmer_list (k,alphabet):
    # Start without values
    return_value = []
    for _ in range(1,k+1):
        return_value.extend(make_kmer_list(_,alphabet))
    
    return(return_value)

##############################################################################
def simple_normalize (vector):
    # Vector type as np.array
    maximum = np.max(vector)
    return_value = vector/maximum
    return(return_value)

##############################################################################
def read_fasta(fasta_file_path):
    sequences = {}
    with open(fasta_file_path, 'r') as file:
        sequence_id = None
        sequence_data = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # Encuentra un encabezado de secuencia
                if sequence_id:
                    # Almacena la secuencia anterior
                    sequences[sequence_id] = ''.join(sequence_data)
                # Reinicia los datos de la secuencia
                sequence_id = line[1:]  # Excluye el '>'
                sequence_data = []
            else:
                # Continua acumulando la secuencia
                sequence_data.append(line)
        # Asegurarse de guardar la última secuencia leída
        if sequence_id:
            sequences[sequence_id] = ''.join(sequence_data)
    return sequences

################################################################################
def split_sequence_in_chunks(sequence, chunk_size):
    # Partition of k length
    return [sequence[i:i+chunk_size] for i in range(0, len(sequence), chunk_size)]

################################################################################
#N: Any base (A, C, G, o T)
#R: A o G (purinas)
#Y: C o T (pirimidinas)
#K: G o T
#M: A o C
#S: G o C
#W: A o T
#B: C, G, o T
#D: A, G, o T
#H: A, C, o T
#V: A, C, o G

def vector_define(sequence, chunk_size):
    kmer_list = make_kmer_list(chunk_size, "ACGTNRYKMSWBDHV")
    
    # k-mers divition
    sequence_kmers = split_sequence_in_chunks(sequence, chunk_size)
    
    # Inicialization vector
    vector = []

    for kmer in sequence_kmers:
        try:
            if len(kmer) == chunk_size:
                vector.append(kmer_list.index(kmer))
        except ValueError:
            print(f"k-mer '{kmer}' not found in the list of possible k-mers.")
            vector.append(-1)  
    return vector
        
        

# Read fasta 
fasta_file_path = "data\Milseq.fasta"
fasta_sequences = read_fasta(fasta_file_path)

# 2 printed
for seq_id in list(fasta_sequences)[:2]:  # Solo los primeros 2 para no ser demasiado extenso
    print(f"ID: {seq_id}")
    print(f"Sequence: {fasta_sequences[seq_id][:60]}...")  # Mostrar solo los primeros 60 caracteres
    print("\n")

ID1 = list(fasta_sequences)[0]
sequence_ID1 = fasta_sequences[ID1][:]
Split = split_sequence_in_chunks(sequence_ID1, 5)
V_ID1 = vector_define(sequence_ID1,5)
N_ID1 = simple_normalize(V_ID1)


k = 3
alphabet = ["A","C","G","T"]
A13 = make_upto_kmer_list([1,2,3],alphabet)
A_3 = make_to_kmer_list(3,alphabet)
Fasta = open(fasta_file_path, 'r')