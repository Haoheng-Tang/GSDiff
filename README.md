# GSDiff
Official implementation of the AAAI 2025 paper: "GSDiff: Synthesizing Vector Floorplans via Geometry-enhanced Structural Graph Generation"

## Data
1. Create folder `datasets/rplandata/Data`.
2. Download the 80,788 RPLAN dataset (http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html). It contains a `floorplan_dataset` folder. Place this `floorplan_dataset` folder under `datasets/rplandata/Data`.
3. Run the following scripts to obtain structural graph data:
   ```bash
   python rplan-extract.py
   python rplan-process1.py
   python rplan-process2.py
   python rplan-process3.py
   python rplan-process4.py
   ```
   - After completion, a directory `rplang-v3-withsemantics` (65,763 train + 3,000 val + 3,000 test = 71,763 `.npy` files) will be created under `datasets/rplandata/Data`.
   - Running `rplan-extract.py` also generates folders (`1-channel-semantics-256`, `3-channel-semantics-256`, `bin_imgs`, `e_imgs`, etc.). You can remove them.

4. Run the following scripts to obtain structural graph data with boundaries:
   ```bash
   python rplan-process5.py
   python rplan-process6.py
   python rplan-process7.py
   ```
   - After completion, 2 directories `rplang-v3-withsemantics-withboundary` and `rplang-v3-withsemantics-withboundary-v2`, will appear under `datasets/rplandata/Data`, each containing 71,763 files.

5. Run the following scripts to obtain topology graphs:
   ```bash
   python rplan-process8.py
   python rplan-process9.py
   python rplan-process10.py
   ```
   - After completion, a `rplang-v3-bubble-diagram` folder will be created under `datasets/rplandata/Data`, containing the same number of files.

6. Move data from `datasets/rplandata/Data` to `datasets` for train/val/testing:
   ```bash
   python move.py
   ```

Note, when we conducted our experiments, the semantics of bubble diagram GT involved randomness. Due to the terms of the RPLAN dataset, we are not permitted to release any part of it. Therefore, the bubble diagram GT semantics extracted using the provided scripts may differ slightly from our experimental results. However, given the large data scale, the bias should be minor (for rooms with ambiguous categories, both our GT and the GT you extract randomly select one category, which is no statistically significant difference).
Alternatively, you can use `get_cycle_basis_and_semantic_3_semansimplified` instead of `get_cycle_basis_and_semantic_2_semansimplified` in `rplan-process8/9/10.py` to extract room semantics and train your own topology models. This method is not random, may yield improvements over the metrics reported in the paper.

--------------------------MSD (Modified Swiss Dwellings)---------------------------------------

The MSD dataset is an alternative to RPLAN. It contains Swiss residential floor plans with richer geometry (room polygons) and 9 room types.

**Room type mapping** (verified by cross-referencing graph node centroids with the raw CSV):

| Code | Room type  |
|------|------------|
| 0    | Bedroom    |
| 1    | Livingroom |
| 2    | Kitchen    |
| 3    | Dining     |
| 4    | Corridor   |
| 5    | Stairs     |
| 6    | Storeroom  |
| 7    | Bathroom   |
| 8    | Balcony    |

**Setup:**
1. Place the MSD data under `datasets/msddata/` with the following structure:
   ```
   datasets/msddata/
   ├── mds_V2_5.372k.csv
   └── modified-swiss-dwellings-v2/
       ├── train/   (graph_in/, graph_out/, full_out/, struct_in/)
       └── test/    (same subdirectories)
   ```
   Each pickle file is named by its `floor_id` from the CSV.

2. Run the preprocessing script to produce GSDiff-compatible `.npy` files:
   ```bash
   python datasets/msd_process.py
   ```
   This creates `datasets/msd-v1-withsemantics/{train,val,test}/` with 3,818 train / 400 val / 727 test samples. Plans with more than 53 rooms are filtered out (~8% of the dataset).

3. Extract topology statistics across the dataset:
   ```bash
   python datasets/msd_topology.py
   ```
   Prints per-split statistics: rooms/plan, edges/plan, average degree, connectivity percentage, and room-type and edge-type distributions.

4. Visualize 5 randomly sampled floor plans as polygon floor plans and adjacency graphs:
   ```bash
   python scripts/plot_msd_floorplans.py
   ```
   Saves to `test_outputs/msd_floorplan_plots/`: one PNG per floor plan (polygon view + graph view) and a combined `overview_5plans.png`. Edit `RANDOM_SEED` and `N_PLANS` at the top of the script to change the sample.

5. Train Stage 1 (node/room generation) on MSD:
   ```bash
   python scripts/trainval_main_msd_unconstrained.py
   ```
   Checkpoints and loss curves are saved under `outputs/msd-stage1-unconstrained/`. The model uses 9 room types (`HeterHouseModel(num_room_types=9)`); the default RPLAN model is unchanged.

6. **Toy MSD run (1000 train / 100 test)** — useful for smoke-testing the pipeline on a small dataset before a full run:

   a. Build the toy folder from the already-processed MSD data. This copies the first 1000 `.npy` files (sorted by numeric ID) from `msd-v1-withsemantics/train/` and the first 100 from `msd-v1-withsemantics/test/` into `datasets/msd-v1-toy/{train,test}/`:
      ```powershell
      $src = 'datasets\msd-v1-withsemantics'; $dst = 'datasets\msd-v1-toy'
      New-Item -ItemType Directory -Force -Path "$dst\train", "$dst\test" | Out-Null
      Get-ChildItem "$src\train" -Filter *.npy | Sort-Object { [int]($_.BaseName) } | Select-Object -First 1000 | ForEach-Object { Copy-Item $_.FullName "$dst\train\" }
      Get-ChildItem "$src\test"  -Filter *.npy | Sort-Object { [int]($_.BaseName) } | Select-Object -First 100  | ForEach-Object { Copy-Item $_.FullName "$dst\test\"  }
      ```
      The `MSDRoomSemantics` dataset class accepts a `data_root='msd-v1-toy'` argument to point at this folder.

   b. Train Stage 1 on the toy set:
      ```bash
      python scripts/trainval_main_msd_toy.py
      ```
      Identical model and pipeline to step 5, but with `total_steps=5000` and the val/FID loop pointed at the toy `test/` split (since the toy folder has no val split). Outputs go to `outputs/msd-stage1-toy/`. Edit `total_steps`, `interval`, `batch_size`, or `device` at the top of the script.

--------------------------LIFULL---------------------------------------

If you want to try training/generating on the LIFULL dataset, please create path `datasets/lifulldata` and follow the data request process of Raster-to-Graph (https://github.com/SizheHu/Raster-to-Graph) to place the data under this path `datasets/lifulldata`. 

The data contains 10,804 images (Step 1: Access the "LIFULL HOME'S Data") and corresponding annotations (Step 2: Access the Annotations).


# Usage
The test scripts for no constraints, topology constraints, and boundary constraints are all placed under `scripts` (test_xxx.py). 
Download the corresponding weights and run them via:
   ```bash
   python test_xxx.py
   ```

No constraints: We use the original 3000 results and run them 5 times to get the average.

To train an unconstrained model on RPLAN, run:

```
python scripts/trainval_simplified_edge_unconstrained.py
```
```
python scripts/trainval_main_unconstrained.py
```


Topology constraints: We took the intersection of the original 3000 results with the test set numbers of HouseDiffusion and House-GAN++, and got 757. 
We ran them 5 times and averaged them to get the FID, KID, GED, and statistical analysis of each room type. 
The sample numbers of 757 are in line 183 of `evalmetric-topoconstrain-ged-roomnumber.py`.

Boundary constraints: We took the intersection of the original 3000 results with the test set numbers of HouseDiffusion and House-GAN++, and got 378. 
We ran them 5 times and averaged them to get the FID, KID, GED, and statistical analysis of each room type. 
The sample number of 378 is on line 9 of `evalmetric-boun-constrain-fid-kid.py`.

--------------------------LIFULL---------------------------------------

All training and testing scripts on LIFULL dataset have 'lifull' in the file names. 

Like RPLAN dataset, the purpose of each script is stated at the top of the script.


# params (place in the 'outputs' folder)
unconstrained params: https://drive.google.com/file/d/15gM0GtW2GwHmlpz0r-rpvo-k-BlNy_gu/view?usp=sharing

topology-constrained params: https://drive.google.com/file/d/1pk7SmvLZ8ON3OUL3SNxPRu73ndVKru0z/view?usp=sharing

boundary-constrained params: https://drive.google.com/file/d/1puqxXIW4Y7AeQHFuC76PlYpWQm6MD8PS/view?usp=sharing

boundary-autoencoder CNN params: https://drive.google.com/file/d/1l6QRpfX5Jtucg3R995HajlwRG8SewUJW/view?usp=sharing

topology-autoencoder Transformer params: https://drive.google.com/file/d/1tExX8LdrFpJfBQH5y2emC6BltBwf9tHx/view?usp=sharing

unconstrained sloping walls params: https://drive.google.com/file/d/1aQNaQwHPkdlNyMQoazZDye7_O8Qbqvzq/view?usp=drive_link

--------------------------LIFULL---------------------------------------

Training parameters on the LIFULL dataset: 

Node: https://drive.google.com/file/d/1k_q9-vQXbs3PDzLxvz-tQRvO3j0DzlPN/view?usp=sharing

Edge: https://drive.google.com/file/d/1XkoMZAMOeBPTteUTVDukgc4BNoEEJSXS/view?usp=sharing
