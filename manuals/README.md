# Bundled owner's manuals

These two owner's manuals are included so the demo works straight after cloning. Both are
published by their manufacturers and given to owners and the public free of charge. The copies
here are byte-for-byte identical to the manufacturers' downloads (SHA-256 below). They are
reproduced unmodified, for reference and to demonstrate manual-grounded answers.

| File | Manual | Copyright | Official source |
|---|---|---|---|
| [`royal-enfield-new-himalayan-450-owners-manual-bs-six-english.pdf`](royal-enfield-new-himalayan-450-owners-manual-bs-six-english.pdf) | Royal Enfield Himalayan 450 (BS6) Owner's Manual, English. Part No. RAM00818/A, 4 November 2025, 149 pages | © Copyright 2023 Royal Enfield (A unit of Eicher Motors Ltd.). All rights reserved. | [royalenfield.com](https://www.royalenfield.com/content/dam/open-pdf/royal-enfield-new-himalayan-450-owners-manual-bs-six-english.pdf) |
| [`TVS-Ronin.pdf`](TVS-Ronin.pdf) | TVS Ronin Owner's Manual, 2026 edition, 134 pages | © TVS Motor Company Limited. All rights reserved. | [tvsmotor.com](https://www.tvsmotor.com/-/media/Feature/Owners/UserManual2026/TVS-Ronin.pdf) |

SHA-256:

```text
cf221a29e2f9cd5af6e80a99774a697c814e8eda661c27d72e519d8a7e80da89  royal-enfield-new-himalayan-450-owners-manual-bs-six-english.pdf
7dc1288d9b3af0b29b94ea96fb32de6be169320eaeb79b5d81e7a19e8d41acfc  TVS-Ronin.pdf
```

- The manuals remain the property of their copyright holders. This repository's MIT license
  covers its code and documentation only, not these manuals.
- The derived index in [`../index/`](../index/) (page text split into passages, plus their
  embeddings) is built from these files and is attributed in the same way.
- Royal Enfield, Himalayan, TVS and Ronin are trademarks of their owners. This project is not
  affiliated with or endorsed by Royal Enfield, Eicher Motors or TVS Motor Company.
- Always use the manufacturer's current manual for your motorcycle; the official links above
  may carry newer editions. `python scripts/fetch_manuals.py --verify-only` checks these copies
  against [`sources.json`](sources.json).
- If you hold the rights to one of these manuals and want it removed, please open an issue
  and it will be taken down.
