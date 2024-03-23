---
layout: post
title: How to represent protein structures in ML
image: /assets/img/blog/prot_representation/protein_bits.png
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  How algorithms such as AlphaFold turn PDB structures into a format that they can process
invert_sidebar: true
---

# How to represent protein structures in ML

Machine Learning approaches empower [a new suite of algorithms and applications](https://www.sciencedirect.com/science/article/abs/pii/S2405471223002983) in structural biology and protein engineering/design. However, there is quite a gap between how protein structure data is classically stored in databases and how machine learning algorithms deal with data. Here, I want to bridge that gap and show how current algorithms such as AlphaFold2 make use of protein structure data in practice.

* toc
{:toc}

## Protein Structure File Formats: PDB vs PDBx/mmCIF vs MMTF vs BinaryCIF

Before we turn to machine learning algorithms such as AlphaFold2, let's shortly discuss how these coordinates are stored in the [PDB](https://www.rcsb.org/) to start with.

Over the years there has been quite an evolution with respect to data formats for protein structures. 

### PDB format (legacy)

The original PDB format [introduced in 1976](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) was intended as a human-readable file that would allow researchers to exchange data easily. While very successful, it is a very wasteful format by today's standards in terms of whitespace and indentation, making automatic parsing realtively difficult. 

Here an excerpt of the PDB file of a [Lysozyme structure](https://www.rcsb.org/structure/168L):

~~~bash
# file: "168l.pdb"
HEADER    HYDROLASE (O-GLYCOSYL)                  24-MAR-95   168L              
TITLE     PROTEIN FLEXIBILITY AND ADAPTABILITY SEEN IN 25 CRYSTAL FORMS OF T4   
TITLE    2 LYSOZYME                                                             
COMPND    MOL_ID: 1;                                                            
COMPND   2 MOLECULE: T4 LYSOZYME;                                               
COMPND   3 CHAIN: A, B, C, D, E;                                                
COMPND   4 EC: 3.2.1.17;                                                        
COMPND   5 ENGINEERED: YES                                                      
SOURCE    MOL_ID: 1;                                                            
SOURCE   2 ORGANISM_SCIENTIFIC: ENTEROBACTERIA PHAGE T4;                        
SOURCE   3 ORGANISM_TAXID: 10665;                                               
SOURCE   4 EXPRESSION_SYSTEM_VECTOR_TYPE: PLASMID;                              
SOURCE   5 EXPRESSION_SYSTEM_PLASMID: M13                                       
KEYWDS    HYDROLASE (O-GLYCOSYL)                                                
EXPDTA    X-RAY DIFFRACTION                                                     
AUTHOR    X.-J.ZHANG,B.W.MATTHEWS                                               
REVDAT   5   07-FEB-24 168L    1       REMARK SEQADV                            
REVDAT   4   29-NOV-17 168L    1       REMARK HELIX                             
REVDAT   3   24-FEB-09 168L    1       VERSN                                    
REVDAT   2   01-APR-03 168L    1       JRNL                                     
REVDAT   1   10-JUL-95 168L    0                                                
JRNL        AUTH   X.J.ZHANG,J.A.WOZNIAK,B.W.MATTHEWS                           
JRNL        TITL   PROTEIN FLEXIBILITY AND ADAPTABILITY SEEN IN 25 CRYSTAL      
JRNL        TITL 2 FORMS OF T4 LYSOZYME.                                        
JRNL        REF    J.MOL.BIOL.                   V. 250   527 1995              
JRNL        REFN                   ISSN 0022-2836                               
JRNL        PMID   7616572                                                      
JRNL        DOI    10.1006/JMBI.1995.0396                                       
REMARK   1                                                                      
REMARK   1 REFERENCE 1                                                          
REMARK   1  AUTH   L.H.WEAVER,B.W.MATTHEWS                                      
REMARK   1  TITL   STRUCTURE OF BACTERIOPHAGE T4 LYSOZYME REFINED AT 1.7        
REMARK   1  TITL 2 ANGSTROMS RESOLUTION                                         
REMARK   1  REF    J.MOL.BIOL.                   V. 193   189 1987              
REMARK   1  REFN                   ISSN 0022-2836                               
REMARK   2                                                                      
REMARK   2 RESOLUTION.    2.90 ANGSTROMS.
...
SEQRES   1 A  164  MET ASN ILE PHE GLU MET LEU ARG ILE ASP GLU GLY LEU          
SEQRES   2 A  164  ARG LEU LYS ILE TYR LYS ASP THR GLU GLY TYR TYR THR          
SEQRES   3 A  164  ILE GLY ILE GLY HIS LEU LEU THR LYS SER PRO SER LEU          
SEQRES   4 A  164  ASN ALA ALA LYS SER GLU LEU ASP LYS ALA ILE GLY ARG          
SEQRES   5 A  164  ASN CYS ASN GLY VAL ILE THR LYS ASP GLU ALA GLU LYS
...
HELIX    1  A1 ILE A    3  GLU A   11  1                                   9    
HELIX    2  A2 LEU A   39  ILE A   50  1                                  12    
HELIX    3  A3 LYS A   60  ARG A   80  1                                  21    
HELIX    4  A4 ALA A   82  SER A   90  1                                   9    
HELIX    5  A5 ALA A   93  MET A  106  1                                  14    
...
ATOM      1  N   MET A   1      74.851  69.339  -6.260  1.00 37.97           N  
ATOM      2  CA  MET A   1      75.137  68.258  -5.357  1.00 38.78           C  
ATOM      3  C   MET A   1      73.896  67.665  -4.750  1.00 40.36           C  
ATOM      4  O   MET A   1      72.862  68.348  -4.627  1.00 40.50           O  
ATOM      5  CB  MET A   1      76.039  68.696  -4.203  1.00 40.16           C      
~~~

You can imagine how parsing something like the resolution automatically from this might be quite a pain. The main structure of such a PDB file is as follows:

- it starts with a `HEADER` and some additional metadata such as the authors and the journal where the structure was published.
- then there are many `REMARKS` that give additional information like the resolution of the structure and the experimental method by which it was acquired.
- what follows is the `SEQRES` (short for sequence representation) that lists the sequence for the structure for quick parsing (more information on this [here](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/primary-sequences-and-the-pdb-format)).
- following is some information about assigned secondary structure indicated via `HELIX` or `SHEET`
- finally, the actual structure information with coordinates etc is prefaced with the `ATOM` qualifier and information such as the atom type described, the residue name, which chain it is part of and of course the coordinates as well as additional metadata such as the [B-factor](https://proteopedia.org/wiki/index.php/Temperature_value).

Two important things to note at this point:
1. Against intuition, the `SEQRES` information does not always align with the sequence contained in the structure itself via the `ATOM` fields. This is a problem that plagues later data formats as well and can be [attributed to a variety of reasons](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/primary-sequences-and-the-pdb-format), mostly that flexible loops and chain ends are often not resolved in experimental structures but still present in the `SEQRES` representation. That is the reason why models like AlphaFold2 and OpenFold require tools like [KAlign](https://academic.oup.com/bioinformatics/article/36/6/1928/5607735) to align the sequence representation to the structure representation in cases where they do not match in the case of template search (see for example [this file](https://github.com/aqlaboratory/openfold/blob/main/openfold/data/tools/kalign.py) in the OpenFold codebase or section 1.2.3 in the [AlphaFold 2 SI](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf) (page 5)).
2. The atom names are not just the chemical elements (C, N, O, ...), but have specific other descriptors depending on where in the amino acid this element occurs (C can be C, CA, CB, CG, ...). How each of these amino acids is named exactly is set in the [PDB Chemical Component Dictionary](https://www.wwpdb.org/data/ccd#pdbechem), but in general you can keep in mind that for many atoms we enumerate them with greek characters after the atom symbol; CG then stands for "Carbon Gamma", i.e the third carbon atom in the chain.

The PDB format does not support Greek characters, so the atom names are translated into the most similar Latin letters:


| Atom name  | Pronunciation | PDB name |
|------------|---------------|----------|
| &alpha;    | alpha         | A        |
| &beta;     | beta          | B        |
| &gamma;    | gamma         | G        |
| &delta;    | delta         | D        |
| &epsilon;  | epsilon       | E        |
| &zeta;     | zeta          | Z        |
| &nu;       | nu            | H        |

C$$\alpha$$ is thus called CA, O$$\gamma$$ is called OG and so on. Sometimes (e.g. in Asp) there may be two identical atoms in the same position, whereby they are named 1 and 2, e.g. the two carboxyl atoms in Asp are called OD1 and OD2. Later in this article we will see a representation of these atoms for all amino acids, but for now we can use the [PDBeChem interface](https://www.ebi.ac.uk/pdbe-srv/pdbechem/) to look up this representation for the amino acid (or, in fact, any chemical component in the PDB) that we are interested in.

If you insert `SER` for the amino acid serine in the "Code" search box, hit the `Search` button and upon getting the result click the `Atoms` tab on the left-hand side of the page, you will get all the atoms in that specific amino acid. We will see later that the representation in models such as AlphaFold2 is a bit shorter since a) they do not include hydrogens in the model and b) one oxygen atom is lost in the condensation of the individual amino acids into the backbone (one water molecule per bond formed to be precise).


### PDBx/mmCIF format

As mentioned, the PDB format has quite some limitations when it comes to supporting large structures as well as complex chemistries. To improve on this, a new format called [PDBx/mmCIF](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/beginner%E2%80%99s-guide-to-pdb-structures-and-the-pdbx-mmcif-format) was introduced and is currently the default format in the PDB. It uses the ASCII character set and is a tabular data format, in which data items have a name of the format `_categoryname.attributename`, for example `_citation_author.name`. If there is only one value for this data item, it is displayed in the same line as a key-value pair. If there are multiple values for these names, a `loop_` token prefaces the categories, followed by rows of data items where the different values are separeted by white spaces.

Compared to the legacy PDB format where a structure is just described as a list of atoms and amino acids, PDBx/mmCIF has more semantics in its representation. One example of this are *entities*, which are defined as [`chemically distinct part of a structure as represented in the PDBx/mmCIF data file`](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/beginner%E2%80%99s-guide-to-pdb-structures-and-the-pdbx-mmcif-format). For example, a chemical ligand would be an entity, or chains in a protein would be an entity. Importantly, these entities can be present multiple times: A homodimer will have one entity since the same chain is present twice.

With this background, let us look at the PDBx/mmCIF file for the same lysozyme structure we looked at before:

~~~bash
# file: "168l.cif"
data_168L
# 
_entry.id   168L 
# 
_audit_conform.dict_name       mmcif_pdbx.dic 
_audit_conform.dict_version    5.385 
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic 
# 
loop_
_database_2.database_id 
_database_2.database_code 
_database_2.pdbx_database_accession 
_database_2.pdbx_DOI 
PDB   168L         pdb_0000168l 10.2210/pdb168l/pdb 
WWPDB D_1000170153 ?            ?                   
# 
...
_entity.id                         1 
_entity.type                       polymer 
_entity.src_method                 man 
_entity.pdbx_description           'T4 LYSOZYME' 
_entity.formula_weight             18373.139 
_entity.pdbx_number_of_molecules   5 
_entity.pdbx_ec                    3.2.1.17 
_entity.pdbx_mutation              ? 
_entity.pdbx_fragment              ? 
_entity.details                    ? 
# 
_entity_poly.entity_id                      1 
_entity_poly.type                           'polypeptide(L)' 
_entity_poly.nstd_linkage                   no 
_entity_poly.nstd_monomer                   no 
_entity_poly.pdbx_seq_one_letter_code       
;MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILR
NAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDAAAAALAAAAWYNQTPNRAKRVITTFRTGTWDA
YKNL
;
_entity_poly.pdbx_seq_one_letter_code_can   
;MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILR
NAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDAAAAALAAAAWYNQTPNRAKRVITTFRTGTWDA
YKNL
;
_entity_poly.pdbx_strand_id                 A,B,C,D,E 
_entity_poly.pdbx_target_identifier         ? 
# 
loop_
_entity_poly_seq.entity_id 
_entity_poly_seq.num 
_entity_poly_seq.mon_id 
_entity_poly_seq.hetero 
1 1   MET n 
1 2   ASN n 
1 3   ILE n 
1 4   PHE n 
1 5   GLU n 
1 6   MET n 
1 7   LEU n 
1 8   ARG n 
...
loop_
_chem_comp.id 
_chem_comp.type 
_chem_comp.mon_nstd_flag 
_chem_comp.name 
_chem_comp.pdbx_synonyms 
_chem_comp.formula 
_chem_comp.formula_weight 
ALA 'L-peptide linking' y ALANINE         ? 'C3 H7 N O2'     89.093  
ARG 'L-peptide linking' y ARGININE        ? 'C6 H15 N4 O2 1' 175.209 
ASN 'L-peptide linking' y ASPARAGINE      ? 'C4 H8 N2 O3'    132.118 
ASP 'L-peptide linking' y 'ASPARTIC ACID' ? 'C4 H7 N O4'     133.103 
CYS 'L-peptide linking' y CYSTEINE        ? 'C3 H7 N O2 S'   121.158 
GLN 'L-peptide linking' y GLUTAMINE       ? 'C5 H10 N2 O3'   146.144 
GLU 'L-peptide linking' y 'GLUTAMIC ACID' ? 'C5 H9 N O4'     147.129 
...
loop_
_atom_site.group_PDB 
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_alt_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_entity_id 
_atom_site.label_seq_id 
_atom_site.pdbx_PDB_ins_code 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z 
_atom_site.occupancy 
_atom_site.B_iso_or_equiv 
_atom_site.pdbx_formal_charge 
_atom_site.auth_seq_id 
_atom_site.auth_comp_id 
_atom_site.auth_asym_id 
_atom_site.auth_atom_id 
_atom_site.pdbx_PDB_model_num 
ATOM 1    N N   . MET A 1 1   ? 74.851  69.339  -6.260  1.00 37.97  ? 1   MET A N   1 
ATOM 2    C CA  . MET A 1 1   ? 75.137  68.258  -5.357  1.00 38.78  ? 1   MET A CA  1 
ATOM 3    C C   . MET A 1 1   ? 73.896  67.665  -4.750  1.00 40.36  ? 1   MET A C   1 
ATOM 4    O O   . MET A 1 1   ? 72.862  68.348  -4.627  1.00 40.50  ? 1   MET A O   1 
ATOM 5    C CB  . MET A 1 1   ? 76.039  68.696  -4.203  1.00 40.16  ? 1   MET A CB  1 
ATOM 6    C CG  . MET A 1 1   ? 76.921  67.555  -3.776  1.00 41.09  ? 1   MET A CG  1 
ATOM 7    S SD  . MET A 1 1   ? 77.902  67.038  -5.191  1.00 40.98  ? 1   MET A SD  1 
ATOM 8    C CE  . MET A 1 1   ? 78.748  65.645  -4.424  1.00 41.39  ? 1   MET A CE  1 
ATOM 9    N N   . ASN A 1 2   ? 74.139  66.409  -4.302  1.00 41.77  ? 2   ASN A N   1 
...
ATOM 6442 C CG  . LEU E 1 164 ? 95.884  25.834  -10.740 0.00 85.05  ? 164 LEU E CG  1 
ATOM 6443 C CD1 . LEU E 1 164 ? 96.110  27.302  -11.107 0.00 85.07  ? 164 LEU E CD1 1 
ATOM 6444 C CD2 . LEU E 1 164 ? 94.874  25.202  -11.694 0.00 85.06  ? 164 LEU E CD2 1 
ATOM 6445 O OXT . LEU E 1 164 ? 98.129  21.647  -9.779  0.00 84.32  ? 164 LEU E OXT 1 
# 
~~~

### MMTF format (legacy)

PDBx/mmCIF is now the standard format for storing macromolecular data. While due to its extensible and verbose format it has rich metadata and is sutied for *archival* purposes, it is not the best format to *transmit* large amounts of structural data due to redundant annotations and repetitive information as you have seen above. Also, the inefficient representation of coordinates separated by whitespaces to make it human-readable is another hurdle for fast transmission of data. 

Due to these limitations, the [MMTF format](https://mmtf.rcsb.org/index.html) (Macromolecular transmission format) was introduced. It does not contain all data present in the PDBx/mmCIF files, but all the data necessary for most visualisation and structural analysis programs. The main pros of MMTF are its compact encoding and fast parsing due to binary instead of string representations. 

![MMTF Compression Pipeline](/assets/img/blog/prot_representation/mmtf_parsing.png)

Overview of the MMTF compression pipeline. Source: [UCSD Presentation](https://github.com/sbl-sdsc/mmtf-workshop-2018/blob/master/0-introduction/MMTF2018-Introduction.pdf)
{:.figcaption}

We can see that after some data preparation, the main step in the MMTF pipeline are various ways of encoding to reduce the file size:
- [integer encoding]()
- [dictionary encoding]()
- [run-length encoding]()
- [delta encoding]()

After these encodings, the file size if comprised further by packing into the [MessagePack format](https://msgpack.org/). It's slogan reads like *It's like JSON, but fast and small*, indicating its flexiblity to store data e.g. as key-value pars, but in a binarized format.

MMTF is great for fast transmission of data and rethought quite a lot of things in clever ways. However, it deviated quite a bit from the mmCIF standard and therefore [never really caught on in the community](https://bioinformatics.stackexchange.com/questions/14738/binarycif-vs-mmtf-formats-which-one-to-choose). This is now finally confirmed with MMTF being [deprecated from July 2024 onward](https://www.mail-archive.com/ccp4bb@jiscmail.ac.uk/msg56121.html).

### BinaryCIF format 

There was a need for a binarized efficient format for protein structure information transfer that was more aligned with the PDBx/mmCIF file format specification. Enter [Binary CIF paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008247), a newer format that is way easiert to interconvert with the now standard PDBx/mmCIF. The [Binary CIF specification](https://github.com/dsehnal/BinaryCIF) is actually quite readable, so I recommend to check it out.

BinaryCIF was heavily inspired by MMTF, with many people working on both formats. This is visible in the usage of MessagePack and the different encodings employed.

![BinaryCIF compressions](/assets/img/blog/prot_representation/binary_cif_compression.png)

Encodings employed for BinaryCIF. Source: [BinaryCIF paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008247) 
{:.figcaption}

There are a few additional ones that you can read up on in the specification on GitHub, but mostly the same encodings as in MMTF were used.

## Coordinates: Atom14 vs Atom37

We discussed how protein structures are stored in databases; with that done, let us talk about how they are represented in machine learning algorithms.

When looking at either the original [AlphaFold codebase](https://github.com/google-deepmind/alphafold) or the open-source reproduction in PyTorch called [OpenFold](https://github.com/aqlaboratory/openfold), many people trip over how the file formats just discussed are represented inside the neural network. This confusion is enhanced by there being two different network-internal representations which are converted into each other depending on the use case scenario.

The documentation on these two representations is sparse, with one being available on a [HuggingFace docstring](https://huggingface.co/spaces/simonduerr/ProteinMPNN/blame/e65166bd70446c6fddcc1581dbc6dac06e7f8dca/alphafold/alphafold/model/all_atom.py):

Generally we employ two different representations for all atom coordinates,
one is atom37 where each heavy atom corresponds to a given position in a 37
dimensional array, This mapping is non amino acid specific, but each slot
corresponds to an atom of a given name, for example slot 12 always corresponds
to 'C delta 1', positions that are not present for a given amino acid are
zeroed out and denoted by a mask.
The other representation we employ is called atom14, this is a more dense way
of representing atoms with 14 slots. Here a given slot will correspond to a
different kind of atom depending on amino acid type, for example slot 5
corresponds to 'N delta 2' for Aspargine, but to 'C delta 1' for Isoleucine.
14 is chosen because it is the maximum number of heavy atoms for any standard
amino acid.
The order of slots can be found in 'residue_constants.residue_atoms'.
Internally the model uses the atom14 representation because it is
computationally more efficient.
The internal atom14 representation is turned into the atom37 at the output of
the network to facilitate easier conversion to existing protein datastructures.
{:.note title="atom14 vs atom37"}

What does this mean in practice? Let's look at the code. When looking at [`residue_constants.residue_atoms`](https://github.com/aqlaboratory/openfold/blob/127f1e7023c380c01330cee45544c23c079babe9/openfold/np/residue_constants.py#L355), we get the following description for the `atom14` representation:

~~~python
# file: "residue_constants.py"
# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
residue_atoms = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "N", "NE1", "O"],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"]}
~~~
`atom14` ordering.
{:.figcaption}

We see that depending on whch amino acid we have present, a certain position in a residue array can represent a different atom (for example, position 3 is `CG2` for Threonine, `CG1` for Valine and `N` for Serine). This makes storing this information very efficient, but can be cumbersome if we need to retrieve the coordinates of a certain atom like `N` from our data structure.

On the other hand, the `atom37` representation has a fixed atom data size for every residue. This ordering can be found in [`residue_constants.atom_types`](https://github.com/aqlaboratory/openfold/blob/127f1e7023c380c01330cee45544c23c079babe9/openfold/np/residue_constants.py#L555):

~~~python
# file: "residue_constants.py"
# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.
~~~

`atom37` ordering.
{:.figcaption}

Here we can see that the ordering is always the same no matter which residue is represented; however, most of the fields will always be empty since the longest amino acid has only 14 atoms. We therefore exchange efficiency vs standardisation, which explains why internally AF2 often uses `atom14`, but when it interfaces to other programs at I/O it often uses `atom37`.

If we think about our example of `Ser` again, we can see how the machine representations map to the actual amino acid (again with the caveat that hydrogens are ommited and the carboxyl oxygen is not counted since in a peptide backbone it will have let as water).

![serine example](/assets/img/blog/prot_representation/serine_repr.jpeg)

| Category |Atom14| Atom37  |
|-----------------|-----------|---------------|
| Memory Requirements |Efficient | Wasteful     |
| Data Layout |Varying Shape | Fixed Shape      |
| Sequence Dependence |Yes| No      |

### Example: Lysozyme atom numbering

Let us now visualise the concepts we looked at so far (atom names and atom representations) with a concrete example, again based on the lysozyme structure with the PDB code `168l`. Install PyMol (either the [commercial](https://pymol.org/) or the [open-source](https://github.com/schrodinger/pymol-open-source?tab=readme-ov-file) version) and open the program.

If you have not used PyMol before, you can either skip this section or look at [this lesson from my Structural Bioinformatics course](https://structural-bioinformatics.netlify.app/blog/proteins/2023-02-01-lesson1/) that goes over this in detail.
{:.note}


Then, execute the following commands via the integrated terminal:

~~~python
fetch 168lA # get first chain of lysozyme assembly
select range, resi 10-15 # select a subset of residues for simplicity
hide everything # hide the whole structure for clarity\
show sticks, range # show stick representation for the selected subset
~~~


![chain example](/assets/img/blog/prot_representation/chain_repr.jpeg)

## Batching: Padded versus sparse

[Padding and Trunction](https://huggingface.co/docs/transformers/main/en/pad_truncation)

[*Sentence pairs were batched together by approximate sequence length*](https://arxiv.org/abs/1706.03762)

[advanced mini-batching](https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html).

![PyG batching](/assets/img/blog/prot_representation/pyg_batching.png)

Advanced mini-batching in Pytorch Geometric. Source: [YouTube](https://www.youtube.com/watch?v=mz9xYNg9Ofs)
{:.figcaption}

## Reference Systems: Local reference frames vs reference-free coordinates vs internal coordinates

## Credits

Thanks a lot to the organisers of the RosettaCon conference, both for making the conference a great experience and for allowing me to post this summary on their website and use their logo for the post on my website.

*[SERP]: Search Engine Results Page
