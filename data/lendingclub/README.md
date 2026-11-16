# Lending Club accepted loans (not committed)

Used by `phase2/p2_step17_lendingclub_oot.py` for the out-of-time test on real, dated lending data.

Place the file here as `accepted_2007_to_2018Q4.csv.gz`.

- Source: the public "Lending Club loan data" release covering 2007 to 2018 Q4, distributed on Kaggle as
  `accepted_2007_to_2018Q4.csv.gz` (dataset "All Lending Club loan data").
- Size 392,582,231 bytes; SHA-256 `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d`.
  The run in this repository used a copy with exactly this checksum.
- 2,260,701 rows. The script reads it in chunks and keeps 36-month individual loans issued 2010-2015 with a final outcome (611,816 rows).
- The data describes real loans. Check the licence and terms on the dataset page before any use beyond research and coursework.
