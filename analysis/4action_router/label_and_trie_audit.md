# Four-Action Online Router Label and Trie Audit

- Integrity: PASS
- Samples: 6,811
- Complete valid routes: 248,804
- Prefix-trie nodes: 5,112,442
- Router parameters: 7,621,638
- Training datasets: GQA, ChartQA, TextVQA
- WeMath Standard/Pro: explicitly excluded from this run
- Executor contract: `d8f524b928fb30ea0bb37c6a9389893adb338d4f91992d85255fdfb9bea283cb`

Multi-valid outgoing actions are retained at every exact prefix; no sample is expanded in proportion to its route count.
