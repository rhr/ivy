import re, sys
from collections import defaultdict
from itertools import izip_longest, ifilter
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from ivy.storage import Storage

email = ""

def batch(iterable, size):
    "take an iterable and return it in chunks (sub-iterables)"
    args = [ iter(iterable) ]*size
    for x in izip_longest(fillvalue=None, *args):
        yield ifilter(None, x)
        
def extract_gbac(s):
    gbac_re = re.compile(r'[A-Z]{1,2}[0-9]{4,7}')
    return gbac_re.findall(s, re.M)

def extract_gene(seq, gene):
    for t in "exon", "gene":
        for x in seq.features:
            if x.type == t:
                v = x.qualifiers.get("gene")
                if v == [gene]:
                    if x.sub_features:
                        s = [ seq[sf.location.start.position:
                                  sf.location.end.position]
                              for sf in x.sub_features ]
                        return reduce(lambda x,y:x+y, s)
                    else:
                        loc = x.location
                        return seq[loc.start.position-10:loc.end.position+10]

def gi2webenv(gilist):
    h = Entrez.esearch(
        db="nucleotide", term=" OR ".join(gilist), usehistory="y",
        retmax=len(gilist)
        )
    d = Entrez.read(h)
    return d["WebEnv"], d["QueryKey"]

def gi2tax(gi):
    global email
    assert email, "set email!"
    Entrez.email = email
    h = Entrez.elink(dbfrom='taxonomy', db='nucleotide', from_uid=gi,
                     LinkName='nucleotide_taxonomy')
    r = Entrez.read(h)[0]
    h.close()
    i = r['LinkSetDb'][0]['Link'][0]['Id']
    h = Entrez.efetch(db='taxonomy', id=i, retmode='xml')
    r = Entrez.read(h)[0]
    h.close()
    return r

def ac2gi(ac):
    global email
    assert email, "set email!"
    Entrez.email = email
    h = Entrez.esearch(db="nucleotide", term=ac, retmax=1)
    d = Entrez.read(h)['IdList'][0]
    h.close()
    return d

def fetch_aclist(aclist):
    global email
    assert email, "set email!"
    Entrez.email = email
    results = {}
    for v in batch(aclist, 100):
        v = list(v)
        h = Entrez.esearch(
            db="nucleotide",
            term=" OR ".join([ "%s[ACCN]" % x for x in v ]),
            usehistory="y"
            )
        d = Entrez.read(h)
        h.close()
        h = Entrez.efetch(db="nucleotide", rettype="gb", retmax=len(v),
                          webenv=d["WebEnv"], query_key=d["QueryKey"])
        seqs = SeqIO.parse(h, "genbank")
        for s in seqs:
            try:
                ac = s.annotations["accessions"][0]
                if ac in aclist:
                    results[ac] = s
            except:
                pass
        h.close()
    return results

def fetch_gilist(gilist, batchsize=1000):
    global email
    assert email, "set email!"
    Entrez.email = email
    results = {}
    for v in batch(gilist, batchsize):
        v = map(str, v)
        h = Entrez.epost(db="nucleotide", id=",".join(v), usehistory="y")
        d = Entrez.read(h)
        h.close()
        h = Entrez.efetch(db="nucleotide", rettype="gb", retmax=len(v),
                          webenv=d["WebEnv"], query_key=d["QueryKey"])
        seqs = SeqIO.parse(h, "genbank")
        for s in seqs:
            try:
                gi = s.annotations["gi"]
                if gi in v:
                    s.id = organism_id(s)
                    results[gi] = s
            except:
                pass
        h.close()
    return results

def organism_id(s):
    org = (s.annotations.get('organism') or '').replace('.', '')
    return '%s_%s' % (org.replace(' ','_'), s.id.split('.')[0])

def fetchseq(gi):
    global email
    assert email, "set email!"
    Entrez.email = email
    h = Entrez.efetch(db="nucleotide", id=str(gi), rettype="gb")
    s = SeqIO.read(h, 'genbank')
    s.id = organism_id(s)
    return s
    
def create_fastas(data, genes):
    fastas = dict([ (g, file(g+".fasta", "w")) for g in genes ])
    for label, seqs in data.items():
        for gene, s in zip(genes, seqs):
            if s and type(s) != str:
                tag = None
                try:
                    tag = "%s_%s" % (label, s.annotations["accessions"][0])
                except:
                    tag = "%s_%s" % (label, s.name)
                if tag:
                    fastas[gene].write(">%s\n%s\n" % (tag, s.seq))
            else:
                sys.stderr.write(("error: not an accession number? "
                                  "%s (%s %s)\n" % (s, label, gene)))

    for f in fastas.values(): f.close()

def merge_fastas(fnames, name="merged"):
    outfile = file(name+".phy", "w")
    gene2len = {}
    d = defaultdict(dict)
    for fn in fnames:
        gene = fn.split(".")[0]
        for rec in SeqIO.parse(file(fn), "fasta"):
            #sp = "_".join(rec.id.split("_")[:2])
            if rec.id.startswith("Pedicularis"):
                sp = rec.id.split("_")[1]
            else:
                sp = rec.id.split("_")[0]
            sp = "_".join(rec.id.split("_")[:-1])
            seq = str(rec.seq)
            d[sp][gene] = seq
            if gene not in gene2len:
                gene2len[gene] = len(seq)

    ntax = len(d)
    nchar = sum(gene2len.values())
    outfile.write("%s %s\n" % (ntax, nchar))
    genes = list(sorted(gene2len.keys()))
    for sp, data in sorted(d.items()):
        s = "".join([ (data.get(gene) or "".join(["?"]*gene2len[gene]))
                      for gene in genes ])
        outfile.write("%s  %s\n" % (sp, s))
    outfile.close()
    parts = file(name+".partitions", "w")
    i = 1
    for g in genes:
        n = gene2len[g]
        parts.write("DNA, %s = %s-%s\n" % (g, i, i+n-1))
        i += n
    parts.close()

def blast_closest(fasta, e=10):
    f = NCBIWWW.qblast("blastn", "nr", fasta, expect=e, hitlist_size=1)
    rec = NCBIXML.read(f)
    d = rec.descriptions[0]
    result = Storage()
    gi = re.findall(r'gi[|]([0-9]+)', d.title) or None
    if gi: result.gi = int(gi[0])
    ac = re.findall(r'gb[|]([^|]+)', d.title) or None
    if ac: result.ac = ac[0].split(".")[0]
    result.title = d.title.split("|")[-1].strip()
    return result

def blast(query, e=10, n=100, entrez_query=""):
    f = NCBIWWW.qblast("blastn", "nr", query, expect=e, hitlist_size=n,
                       entrez_query=entrez_query)
    rec = NCBIXML.read(f)
    v = []
    for d in rec.descriptions:
        result = Storage()
        gi = re.findall(r'gi[|]([0-9]+)', d.title) or None
        if gi: result.gi = int(gi[0])
        ac = re.findall(r'gb[|]([^|]+)', d.title) or None
        if ac: result.ac = ac[0].split(".")[0]
        result.title = d.title.split("|")[-1].strip()
        v.append(result)
    return v

def start_codons(seq):
    i = seq.find('ATG')
    while i != -1:
        yield i
        i = seq.find('ATG', i+3)

def search_taxonomy(q):
    global email
    assert email, "set email!"
    Entrez.email = email
    h = Entrez.esearch(db="taxonomy", term=q)
    return Entrez.read(h)['IdList']

def fetchtax(taxid):
    global email
    assert email, "set email!"
    Entrez.email = email
    h = Entrez.efetch(db='taxonomy', id=taxid, retmode='xml')
    r = Entrez.read(h)[0]
    return r

__FIRST = re.compile('[^-]')
__LAST = re.compile('[-]*$')
def trimpos(rec):
    'return the positions of the first and last ungapped base'
    s = rec.seq.tostring()
    first = __FIRST.search(s).start()
    last = __LAST.search(s).start()-1
    return (first, last)
