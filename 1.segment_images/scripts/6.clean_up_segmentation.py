#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import shutil

import numpy as np
import tqdm

# In[2]:


overwrite = False


# In[3]:


processed_data_dir = pathlib.Path("../processed_data").resolve(strict=True)
normalized_data_dir = pathlib.Path("../../data/normalized_z").resolve(strict=True)
# normalized_data_dir = pathlib.Path("../../data/test_dir").resolve(strict=True)
cellprofiler_dir = pathlib.Path("../../data/cellprofiler").resolve()
cellprofiler_dir.mkdir(parents=True, exist_ok=True)

# get the list of dirs in the normalized_data_dir
norm_dirs = [x for x in normalized_data_dir.iterdir() if x.is_dir()]
# copy each dir and files to cellprofiler_dir
for norm_dir in tqdm.tqdm(norm_dirs):
    dest_dir = pathlib.Path(cellprofiler_dir, norm_dir.name)
    if dest_dir.exists() and overwrite:
        shutil.rmtree(dest_dir)
        shutil.copytree(norm_dir, dest_dir)
    elif not dest_dir.exists():
        shutil.copytree(norm_dir, dest_dir)
    else:
        pass


# get a list of dirs in processed_data
dirs = [x for x in processed_data_dir.iterdir() if x.is_dir()]


# In[4]:


file_extensions = {".tif", ".tiff"}
# get a list of files in each dir
for well_dir in tqdm.tqdm(dirs):
    files = [x for x in well_dir.iterdir() if x.is_file()]
    for file in files:
        # remove numpy files
        if file.suffix == ".npy":
            file.unlink()
        elif file.suffix in file_extensions:
            new_file_dir = pathlib.Path(
                cellprofiler_dir, well_dir.name, file.stem + file.suffix
            )
            shutil.copy(file, new_file_dir)


# In[5]:


dirs_in_cellprofiler_dir = [x for x in cellprofiler_dir.iterdir() if x.is_dir()]
for dir in tqdm.tqdm(dirs_in_cellprofiler_dir):
    files = [x for x in dir.iterdir() if x.is_file()]
    if len(files) > 8:
        print(f"{dir.name} has too many files: {len(files)}")
    elif len(files) < 8:
        print(f"{dir.name} has too few files: {len(files)}")
    else:
        pass
