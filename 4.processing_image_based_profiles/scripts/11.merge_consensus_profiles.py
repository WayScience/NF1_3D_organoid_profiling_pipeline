#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile aggregation.

# In[1]:


import os
import pathlib

import pandas as pd
from arg_parsing_utils import parse_args
from notebook_init_utils import bandicoot_check, init_notebook
from pycytominer import aggregate

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)


# In[ ]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0014_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


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
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.feature_selected_profiles/organoid_fs.parquet"
).resolve(strict=True)
sc_agg_well_parent_organoid_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.aggregated_profiles/sc_agg_parent_organoid_level.parquet"
).resolve(strict=True)

# output merged path
organoid_agg_well_parent_organoid_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.merged_profiles/sc-organoid_sc_agg_well_parent_organoid_level.parquet"
).resolve()

########################################################################

# 2. The well level profiles are merged together
sc_agg_well_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.aggregated_profiles/sc_agg_well_level.parquet"
).resolve(strict=True)

organoid_agg_well_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.aggregated_profiles/organoid_agg_well_level.parquet"
).resolve(strict=True)

# output merged path
organoid_agg_well_merge_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.merged_profiles/sc-organoid_agg_well_level.parquet"
).resolve()

###################################################################################

# 3. The consensus profiles are merged together

sc_consensus_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.aggregated_profiles/sc_consensus.parquet"
).resolve(strict=True)

organoid_consensus_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.aggregated_profiles/organoid_consensus.parquet"
).resolve(strict=True)

# output merged path
organoid_consensus_merge_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.merged_profiles/sc-organoid_consensus.parquet"
).resolve()


organoid_consensus_merge_path.parent.mkdir(parents=True, exist_ok=True)

###############################################################################


# In[4]:


organoid_fs = pd.read_parquet(organoid_fs_path)
sc_agg_well_parent_organoid = pd.read_parquet(sc_agg_well_parent_organoid_path)
sc_agg_well_parent_organoid_merge = sc_agg_well_parent_organoid.merge(
    organoid_fs,
    left_on=[
        "Metadata_Well",
        "Metadata_parent_organoid",
        "Metadata_treatment",
        "Metadata_Target",
        "Metadata_Class",
        "Metadata_Therapeutic_Categories",
    ],
    right_on=[
        "Metadata_Well",
        "Metadata_object_id",
        "Metadata_treatment",
        "Metadata_Target",
        "Metadata_Class",
        "Metadata_Therapeutic_Categories",
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
    on=[
        "Metadata_Well",
        "Metadata_treatment",
        "Metadata_Target",
        "Metadata_Class",
        "Metadata_Therapeutic_Categories",
    ],
)
sc_agg_well_merge.to_parquet(organoid_agg_well_merge_path, index=False)
sc_agg_well_merge.head()


# In[6]:


sc_consensus = pd.read_parquet(sc_consensus_path)
organoid_consensus = pd.read_parquet(organoid_consensus_path)
sc_consensus_merge = sc_consensus.merge(
    organoid_consensus,
    on=[
        "Metadata_treatment",
        "Metadata_Target",
        "Metadata_Class",
        "Metadata_Therapeutic_Categories",
    ],
)
sc_consensus_merge.to_parquet(organoid_consensus_merge_path, index=False)
sc_consensus_merge.head()
