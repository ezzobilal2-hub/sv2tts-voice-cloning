# Relative default for callers that import this setting. The CLI accepts a dataset root.
datasets_root = "VCTK-Corpus"

# VCTK dataset configuration with train / test / dev splits
vctk_datasets = {
    "train": {"clean": ["VCTK-Corpus/wav48/train"]},
    "test":  {"clean": ["VCTK-Corpus/wav48/test"]},
    "dev":   {"clean": ["VCTK-Corpus/wav48/dev"]},
}

# Disable other datasets (you only need VCTK)
librispeech_datasets = {}
libritts_datasets = {}
voxceleb_datasets = {}

# Other datasets list (scripts loop through this sometimes)
other_datasets = [
    "VCTK-Corpus/wav48/train",
    "VCTK-Corpus/wav48/test",
    "VCTK-Corpus/wav48/dev",
]

# Keep as-is (used in some scripts)
anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]
