# qcm-analysis
This repo contains python source for visualising [Quranic Arabic Corpus](http://corpus.quran.com) (GPL below) in graph topologies using [networkx package](https://networkx.github.io/). This uses v0.4 of the Quran Corpus Morphology.


```#  Quranic Arabic Corpus (Version 0.4)
#  Copyright (C) 2011 Kais Dukes
#  License: GNU General Public License
#
#  The Quranic Arabic Corpus includes syntactic and morphological
#  annotation of the Quran, and builds on the verified Arabic text
#  distributed by the Tanzil project.
#
#  TERMS OF USE:
#
#  - Permission is granted to copy and distribute verbatim copies
#    of this file, but CHANGING IT IS NOT ALLOWED.
#
#  - This annotation can be used in any website or application,
#    provided its source (the Quranic Arabic Corpus) is clearly
#    indicated, and a link is made to http://corpus.quran.com to enable
#    users to keep track of changes.
#
#  - This copyright notice shall be included in all verbatim copies
#    of the text, and shall be reproduced appropriately in all works
#    derived from or containing substantial portion of this file.
#
#  Please check updates at: http://corpus.quran.com/download
```

## Packages required:

```Anaconda3``` is required so might be best to update your distro (if using ```anaconda2```) before installing the following packages (not included in Anaconda).

1. ```python-bidi (bidialg)``` - RTL for arabib display. Install using 'easy_install python-bidi'
2. ```arabic reshaper``` - RTL for arabic display. Install using 'pip install arabic-reshaper'

I use ```Sublime 3``` with Conda integration to code, but any other IDE should be fine.

## qur_corpus_morphology.py
Run this script to create **quran-morphology-final.csv** which is then consumed by other scripts for root analysis

This script loads and transforms the pandas dataframe for Quranic Corups Morphology. It takes snippets from AbdulBaqi Muhammad (Sharaf)'s [blog](http://abdulbaqi.io/2018/12/04/makki-madani-word-count/) and website: [Text Mining the Quran](http://textminingthequran.com/)

## quran_root_analysis.py
Run this script to create graph topologies for any selected Sura, or the entire text of the Quran. Individual root of interest may also be selected, for which subgraphs are created. 

This script visualises root data in terms of frequence or occurrence, its relationships with other roots in a specific sura or the whole Quran in terms of cooccurrence or paired frequency.

~~TODO~~
~~1. Graph properties like ...~~
