# Stage-2 V1 revised frozen protocol

- Contract: `83d647b2e4b2fe09a0cec2a1c0076a6397c832806d8646bdde8599c352d409eb`
- Phase-56 source: `6489a0b39af3efb22bcb527d3b43c474dc7bcb01cc453445aab45f420eb9905b`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct` at `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Trainable component: one shared READ/WRITE router only; no layer identity or Stage-1 latent input.
- State source: exact current routed token sequences replayed online from the selected successful route.
- Data: Corpus A (39 C) + Corpus B (698 W); Corpus C is excluded.
- Sampler: each W UID once/epoch, one uniform successful route, positive-anchored Random-4; 350 C draws/epoch.
- Optimization: 12 epochs, 3,144 global updates, AdamW 3e-4, constant schedule, plain CE.
- Selection: only the final global-update checkpoint; training diagnostics cannot alter it.
- Validation: one frozen 800-sample validation rollout; 235 Stage-1-triggered samples execute Stage-2.
- Stop: after V1 validation/diagnosis; no V1.5 or test execution.
