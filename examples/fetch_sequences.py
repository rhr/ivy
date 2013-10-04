import logging
logging.basicConfig(level=logging.INFO)

from ivy import genbank
genbank.email = 'me@my.address.com'

# Entrez search terms are combined with OR (a OR b OR c OR ...)
terms = ['"Pedicularis rex"[organism]', 'Phtheirospermum[organism]']
seqs = genbank.fetch_DNA_seqs(terms)
with open('myseqs.fasta', 'w') as f:
    genbank.SeqIO.write(seqs, f, 'fasta')
