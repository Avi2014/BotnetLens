# Data Quality Report — CTU-13

**Raw rows:** 19,976,700
**Clean rows:** 19,976,700
**Rows removed:** 0

## Cleaning Steps

- **remove_negative_bytes_pkts**: removed 0 rows (negative byte or packet count)
- **remove_negative_duration**: removed 0 rows (negative duration)
- **remove_unresolved_labels**: removed 0 rows (binary_label == -1)
- **drop_duplicates**: removed 0 rows (exact duplicate rows within chunk)

## Clean Dataset Label Distribution

- Benign: 19,532,001
- Botnet: 444,699

## Family Distribution

- Benign: 19,532,001
- Menti: 4,630
- Murlo: 6,127
- Neris: 246,889
- NsisAy: 2,168
- Rbot: 143,918
- Sogou: 63
- Virut: 40,904