# Online Persistent Corrective Supervision

- Checkpoints executed: 20
- Selected epoch under C2C >= 95%: 14
- Pareto frontier: [{'c2c_preservation_rate': 0.65625, 'epoch': 4, 'w2c_rescue_rate': 0.1640625}, {'c2c_preservation_rate': 0.921875, 'epoch': 12, 'w2c_rescue_rate': 0.1015625}, {'c2c_preservation_rate': 0.953125, 'epoch': 14, 'w2c_rescue_rate': 0.046875}, {'c2c_preservation_rate': 0.9765625, 'epoch': 19, 'w2c_rescue_rate': 0.0390625}, {'c2c_preservation_rate': 0.9765625, 'epoch': 20, 'w2c_rescue_rate': 0.0390625}]
- External evaluation started: false

## Selected-checkpoint dataset behavior

- gqa: W2C 1/43 (0.023256); C2C 41/43 (0.953488)
- chartqa: W2C 1/43 (0.023256); C2C 40/43 (0.930233)
- textvqa: W2C 4/42 (0.095238); C2C 41/42 (0.976190)

## Every checkpoint

- epoch 1: W2C 0.039062; C2C 0.210938; net -96
- epoch 2: W2C 0.039062; C2C 0.109375; net -109
- epoch 3: W2C 0.101562; C2C 0.757812; net -18
- epoch 4: W2C 0.164062; C2C 0.656250; net -23
- epoch 5: W2C 0.062500; C2C 0.601562; net -43
- epoch 6: W2C 0.085938; C2C 0.406250; net -65
- epoch 7: W2C 0.078125; C2C 0.695312; net -29
- epoch 8: W2C 0.046875; C2C 0.851562; net -13
- epoch 9: W2C 0.101562; C2C 0.804688; net -12
- epoch 10: W2C 0.070312; C2C 0.882812; net -6
- epoch 11: W2C 0.054688; C2C 0.843750; net -13
- epoch 12: W2C 0.101562; C2C 0.921875; net 3
- epoch 13: W2C 0.062500; C2C 0.835938; net -13
- epoch 14: W2C 0.046875; C2C 0.953125; net 0
- epoch 15: W2C 0.023438; C2C 0.960938; net -2
- epoch 16: W2C 0.031250; C2C 0.976562; net 1
- epoch 17: W2C 0.039062; C2C 0.968750; net 1
- epoch 18: W2C 0.039062; C2C 0.968750; net 1
- epoch 19: W2C 0.039062; C2C 0.976562; net 2
- epoch 20: W2C 0.039062; C2C 0.976562; net 2
