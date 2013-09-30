from urllib2 import urlopen
from lxml import etree
from collections import defaultdict
from storage import Storage
import sys, re

# "http://purl.org/phylo/treebase/phylows/study/TB2:S11152"

TREEBASE_WEBSERVICE = "http://purl.org/phylo/treebase/phylows"
NEXML_NAMESPACE = "http://www.nexml.org/2009"
NEXML = "{%s}" % NEXML_NAMESPACE
UNIPROT = "http://purl.uniprot.org/taxonomy/"
NAMEBANK = ("http://www.ubio.org/authority/metadata.php?"
            "lsid=urn:lsid:ubio.org:namebank:")

ROW_SEGMENTS = ("http://treebase.org/treebase-web/search/study/"
                "rowSegmentsTSV.html?matrixid=")

META_DATATYPE = {
    "xsd:long": int,
    "xsd:integer": int,
    "xsd:string": str
    }

AMBIG_RE = re.compile(r'([{][a-zA-Z]+[}])')

def fetch_study(study_id, format="nexml"):
    try: study_id = "S%s" % int(study_id)
    except ValueError: pass
        
    # format is one of ["rdf", "html", "nexml", "nexus"]
    url = "%s/study/TB2:%s?format=%s" % (TREEBASE_WEBSERVICE, study_id, format)
    if format=="nexus":
        return urlopen(url).read()
    else:
        return etree.parse(url)

def parse_chars(e, otus):
    v = []
    for chars in e.findall(NEXML+"characters"):
        c = Storage(chars.attrib)
        c.states = parse_states(chars)
        c.meta = Storage()
        for meta in chars.findall(NEXML+"meta"):
            a = meta.attrib
            if a.get("content"):
                value = META_DATATYPE[a["datatype"]](a["content"])
                c.meta[a["property"]] = value
        c.matrices = []
        for matrix in chars.findall(NEXML+"matrix"):
            m = Storage()
            m.rows = []
            for row in matrix.findall(NEXML+"row"):
                r = Storage(row.attrib)
                r.otu = otus[r.otu]
                s = row.findall(NEXML+"seq")[0].text
                substrs = []
                for ss in AMBIG_RE.split(s):
                    if ss.startswith("{"):
                        key = frozenset(ss[1:-1])
                        val = c.states.states2symb.get(key)
                        if key and not val:
                            sys.stderr.write("missing ambig symbol for %s\n" %
                                             "".join(sorted(key)))
                        ss = val or "?"
                    substrs.append(ss)
                s = "".join(substrs)
                r.seq = s
                m.rows.append(r)
            c.matrices.append(m)
        v.append(c)
    return v

def parse_trees(e, otus):
    "e is a nexml document parsed by etree"
    from tree import Node
    v = []
    for tb in e.findall(NEXML+"trees"):
        for te in tb.findall(NEXML+"tree"):
            t = Storage()
            t.attrib = Storage(te.attrib)
            t.nodes = {}
            for n in te.findall(NEXML+"node"):
                node = Node()
                if n.attrib.get("otu"):
                    node.isleaf = True
                    node.otu = otus[n.attrib["otu"]]
                    node.label = node.otu.label
                t.nodes[n.attrib["id"]] = node
            for edge in te.findall(NEXML+"edge"):
                d = edge.attrib
                n = t.nodes[d["target"]]
                p = t.nodes[d["source"]]
                length = d.get("length")
                if length:
                    n.length = float(length)
                p.add_child(n)
            r = [ n for n in t.nodes.values() if not n.parent ]
            assert len(r)==1
            r = r[0]
            r.isroot = True
            for i, n in enumerate(r): n.id = i+1
            t.root = r
            v.append(t)
    return v

def parse_otus(e):
    "e is a nexml document parsed by etree"
    v = {}
    for otus in e.findall(NEXML+"otus"):
        for x in otus.findall(NEXML+"otu"):
            otu = Storage()
            otu.id = x.attrib["id"]
            otu.label = x.attrib["label"]
            for meta in x.iterchildren():
                d = meta.attrib
                p = d.get("property")
                if p and p == "tb:identifier.taxon":
                    otu.tb_taxid = d["content"]
                elif p and p == "tb:identifier.taxonVariant":
                    otu.tb_taxid_variant = d["content"]
                h = d.get("href")
                if h and h.startswith(NAMEBANK):
                    otu.namebank_id = int(h.replace(NAMEBANK, ""))
                elif h and h.startswith(UNIPROT):
                    otu.ncbi_taxid = int(h.replace(UNIPROT, ""))
            v[otu.id] = otu
    return v

def parse_nexml(doc):
    if not isinstance(doc, (etree._ElementTree, etree._Element)):
        doc = etree.parse(doc)
    meta = {}
    for child in doc.findall(NEXML+"meta"):
        if "content" in child.attrib:
            d = child.attrib
            key = d["property"]
            val = META_DATATYPE[d["datatype"]](d["content"])
            if (key in meta) and val:
                if isinstance(meta[key], list):
                    meta[key].append(val)
                else:
                    meta[key] = [meta[key], val]
            else:
                meta[key] = val
            
    otus = parse_otus(doc)

    return Storage(meta = meta,
                   otus = otus,
                   chars = parse_chars(doc, otus),
                   trees = parse_trees(doc, otus))

def parse_states(e):
    "e is a characters element"
    f = e.findall(NEXML+"format")[0]
    sts = f.findall(NEXML+"states")[0]
    states2symb = {}
    symb2states = {}
    id2symb = {}
    for child in sts.iterchildren():
        t = child.tag 
        if t == NEXML+"state":
            k = child.attrib["id"]
            v = child.attrib["symbol"]
            id2symb[k] = v
            states2symb[v] = v
            symb2states[v] = v
        elif t == NEXML+"uncertain_state_set":
            k = child.attrib["id"]
            v = child.attrib["symbol"]
            id2symb[k] = v
            memberstates = []
            for memb in child.findall(NEXML+"member"):
                sid = memb.attrib["state"]
                symb = id2symb[sid]
                for x in symb2states[symb]: memberstates.append(x)
            memberstates = frozenset(memberstates)
            symb2states[v] = memberstates
            states2symb[memberstates] = v
    return Storage(states2symb=states2symb,
                   symb2states=symb2states,
                   id2symb=id2symb)

def parse_charsets(study_id):
    from cStringIO import StringIO
    nx = StringIO(fetch_study(study_id, 'nexus'))
    d = {}
    for line in nx.readlines():
        if line.strip().startswith("CHARSET "):
            v = line.strip().split()
            label = v[1]
            first, last = map(int, line.split()[-1][:-1].split("-"))
            d[label] = (first-1, last-1)
    return d
            
if __name__ == "__main__":
    import sys
    from pprint import pprint
    e = fetch_study('S11152', 'nexus')
    #print e
    #e.write(sys.stdout, pretty_print=True)
    
    ## e = etree.parse('/tmp/tmp.xml')
    ## x = parse_nexml(e)
    ## pprint(x)
    
    
