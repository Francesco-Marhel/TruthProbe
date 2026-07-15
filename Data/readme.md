The calculated truth directions and extracted semantic weights are organized in the `Data/` directory, divided by model architecture and K. 
K-folder contains both the raw PyTorch weights (`.pt`) for direct implementation and human-readable text mappings (`.json`) for quick inspection.

`visualizza_dizionario.py` generates images and gifs to visualize the matrices.
`crea_dizionario.py` It is a single file capable of performing matrix calculations from scratch (`truth_probe.py`), one file, a few commands, identical results (no assembly line).
Timelapse its the same metric misured for multiple layers, in a single gif.


    DATA---|---REPRODUCING-------src
                      |
                      |
                      |---Llama3B/ K8  |  K33 / (.json - .pt) 
                      |             
                      |             
                      |
                      |---Qwen3B/ K8  |  K33 /  (.json - .pt)
