---
layout: post
title: How to represent protein structures in ML
image: /assets/img/blog/rosetta_logo.png
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
- what follows is the `SEQRES` (short for sequence representation) that lists the sequence for the structure for quick parsing.
- following is some information about assigned secondary structure indicated via `HELIX` or `SHEET`
- finally, the actual structure information with coordinates etc is prefaced with the `ATOM` qualifier and information such as the atom type described, the residue name, which chain it is part of and of course the coordinates as well as additional metadata such as the [B-factor](https://proteopedia.org/wiki/index.php/Temperature_value).

Two important things to note at this point:
1. Against intuition, the `SEQRES` information does not always align with the sequence contained in the structure itself via the `ATOM` fields. This is a problem that plagues later data formats as well and can be [attributed to a variety of reasons](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/primary-sequences-and-the-pdb-format), mostly that flexible loops and chain ends are often not resolved in experimental structures but still present in the `SEQRES` representation. That is the reason why models like AlphaFold2 and OpenFold require tools like [KAlign](https://academic.oup.com/bioinformatics/article/36/6/1928/5607735) to align the sequence representation to the structure representation in cases where they do not match in the case of template search (see for example [this file](https://github.com/aqlaboratory/openfold/blob/main/openfold/data/tools/kalign.py) in the OpenFold codebase or section 1.2.3 in the [AlphaFold 2 SI](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf) (page 5)).
2. The atom names are not just the chemical elements (C, N, O, ...), but have specific other descriptors depending on where in the amino acid this element occurs (C can be C, CA, CB, CG, ...).

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

C$$\alpha$$ is thus called CA, O$$\gamma$$ is called OG and so on. Sometimes (e.g. in Asp) there may be two identical atoms in the same position, whereby they are named 1 and 2, e.g. the two carboxyl atoms in Asp are called OD1 and OD2.


### PDBx/mmCIF format

### MMTF format (legacy)

### BinaryCIF format 

## Coordinates: Atom14 vs Atom37

When looking at either the original [AlphaFold codebase]() or the open-source reproduction in PyTorch called [OpenFold](), many people trip over how the file formats just discussed are represented inside the neural network. This confusion is enhanced by there being two different network-internal representations which are converted into each other depending on the use case scenario.

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
{:.note title="Atom14 vs Atom37"}

What does this mean in practice? Let's look at the code. When looking at [`residue_constants.residue_atoms`](https://github.com/aqlaboratory/openfold/blob/127f1e7023c380c01330cee45544c23c079babe9/openfold/np/residue_constants.py#L355), we get the following description:

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
An optional caption for a code block
{:.figcaption}

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
An optional caption for a code block
{:.figcaption}

## Batching: Padded versus sparse


## Reference Systems: Local reference frames vs reference-free coordinates

## Credits

Thanks a lot to the organisers of the RosettaCon conference, both for making the conference a great experience and for allowing me to post this summary on their website and use their logo for the post on my website.

*[SERP]: Search Engine Results Page
