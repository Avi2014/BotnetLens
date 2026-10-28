# Dataset placement

Place **only CTU-13-derived** `.binetflow`, `.csv`, or `.zeek`/flow CSV files under `data/raw/`. The pipeline does not download data and does not bundle CTU-13. Preserve the original scenario folders and filenames where possible. Do not place IoT-23, N-BaIoT, CICIDS, UNSW-NB15, Mirai datasets, or unrelated data here.

The original CTU-13 collection is from 2011 and its scenarios cover families such as Neris, Rbot, Virut, Menti, Sogou, Murlo and NSIS.ay; it is **not a Mirai-labelled dataset**. Family inference from labels alone may be impossible; the code preserves unresolved labels rather than guessing.
