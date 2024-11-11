# Setups

10 epochs, 224 train (GSV), 332 val (MSLS), 128 BS

teacher: DINOv2 w/ GeM, MSLS: 91.0

student: MambaVision w/ GeM w/ Large LR, MSLS: 84.2

student w/ teacher (LoRD): MambaVision w/ GeML w/ Large LR, MSLS: 80.7