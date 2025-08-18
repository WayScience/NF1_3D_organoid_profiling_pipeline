#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile aggregation.

# In[1]:


import argparse
import pathlib

import pandas as pd
from pycytominer import aggregate

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID to process, e.g. 'P01'",
    )
    args = argparser.parse_args()
    patient = args.patient

else:
    patient = "SARCO361"


# ### Merge the sc and organoid profiles after aggregation
# 1. The single-cell parent organoid aggregated profile is merged with the fs organoid profile
# 2. The well level profiles are merged together
# 3. The consensus profiles are merged together
#

# In[3]:


# group the import paths by the type of aggregation
#######################################################################
# 1. The single-cell parent organoid aggregated profile is merged with the fs organoid profile
organoid_fs_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/4.feature_selected_profiles/organoid_fs.parquet"
).resolve(strict=True)
sc_agg_well_parent_organoid_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/sc_agg_parent_organoid_level.parquet"
).resolve(strict=True)

# output merged path
organoid_agg_well_parent_organoid_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/6.merged_profiles/sc-organoid_sc_agg_well_parent_organoid_level.parquet"
).resolve()

########################################################################

# 2. The well level profiles are merged together
sc_agg_well_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/sc_agg_well_level.parquet"
).resolve(strict=True)

organoid_agg_well_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/organoid_agg_well_level.parquet"
).resolve(strict=True)

# output merged path
organoid_agg_well_merge_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/6.merged_profiles/sc-organoid_agg_well_level.parquet"
).resolve()

###################################################################################

# 3. The consensus profiles are merged together

sc_consensus_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/sc_consensus.parquet"
).resolve(strict=True)

organoid_consensus_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/organoid_consensus.parquet"
).resolve(strict=True)

# output merged path
organoid_consensus_merge_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/6.merged_profiles/sc-organoid_consensus.parquet"
).resolve()


organoid_consensus_merge_path.parent.mkdir(parents=True, exist_ok=True)

###############################################################################


# In[4]:


organoid_fs = pd.read_parquet(organoid_fs_path)
sc_agg_well_parent_organoid = pd.read_parquet(sc_agg_well_parent_organoid_path)
sc_agg_well_parent_organoid_merge = sc_agg_well_parent_organoid.merge(
    organoid_fs,
    left_on=[
        "Well",
        "parent_organoid",
        "treatment",
        "Target",
        "Class",
        "Therapeutic Categories",
    ],
    right_on=[
        "Well",
        "object_id",
        "treatment",
        "Target",
        "Class",
        "Therapeutic Categories",
    ],
)

sc_agg_well_parent_organoid_merge.to_parquet(
    organoid_agg_well_parent_organoid_path, index=False
)
sc_agg_well_parent_organoid_merge.head()


# In[5]:


sc_agg_well = pd.read_parquet(sc_agg_well_path)
organoid_agg_well = pd.read_parquet(organoid_agg_well_path)
sc_agg_well_merge = sc_agg_well.merge(
    organoid_agg_well,
    on=["Well", "treatment", "Target", "Class", "Therapeutic Categories"],
)
sc_agg_well_merge.to_parquet(organoid_agg_well_merge_path, index=False)
sc_agg_well_merge.head()


# In[6]:


sc_consensus = pd.read_parquet(sc_consensus_path)
organoid_consensus = pd.read_parquet(organoid_consensus_path)
sc_consensus_merge = sc_consensus.merge(
    organoid_consensus, on=["treatment", "Target", "Class", "Therapeutic Categories"]
)
sc_consensus_merge.to_parquet(organoid_consensus_merge_path, index=False)
sc_consensus_merge.head()
