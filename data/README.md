Download the dataset from [here](https://drive.google.com/drive/folders/1mt36VY2LZRMjpT89__bh_PGsJE3q4RRN?usp=drive_link) and place the contents in the `data/` directory.

Unzip the downloaded `BIM Dataset V3.zip` file so that the directory structure looks like this:

```bash
wqf7006-msl/
└── data/
    └── BIM Dataset V3/
        ├── video/
        │   ├── abang/
        │   │   ├── abang_1_1_1.mp4
        │   │   ├── ...
        │   │   └── abang_4_6_3.mp4
        │   ├── ada/
        │   ├── ...
        │   └── ambil/
        │
        ├── features/ (we only save top 30 glosses with most videos)
        │   ├── first_30/ (first 30 frames of each video)
        │   │   ├── abang/
        │   │   │   ├── abang_1_1_1.mp4/
        │   │   │   │   ├── 00.npy (258-dim landmark features)
        │   │   │   │   ├── ...
        │   │   │   │   └── 29.npy
        │   │   │   └── abang_4_6_3.mp4/
        │   │   ├── ada/
        │   │   ├── ...
        │   │   └── ambil/
        │   │
        │   └── uniform_30/ (uniformly sampled 30 frames from each video)
        │       ├── abang/
        │       │   ├── abang_1_1_1.mp4/
        │       │   │   ├── 00.npy
        │       │   │   ├── ...
        │       │   │   └── 29.npy
        │       │   └── abang_4_6_3.mp4/
        │       ├── ada/
        │       └── ambil/
        │
        └── tensors/
            ├── first_30/
            │   ├── X.npy
            │   ├── y.npy
            │   ├── X_train.npy
            │   ├── y_train.npy
            │   ├── X_test.npy
            │   ├── y_test.npy
            │   └── label_map.json
            │
            └── uniform_30/
                ├── X.npy
                ├── y.npy
                ├── X_train.npy
                ├── y_train.npy
                ├── X_test.npy
                ├── y_test.npy
                └── label_map.json
```
