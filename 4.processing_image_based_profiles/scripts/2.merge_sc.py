#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import duckdb
import pandas as pd
from arg_parsing_utils import parse_args
from cytotable import convert, presets
from notebook_init_utils import bandicoot_check, init_notebook
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


if not in_notebook:
    args = parse_args()
    well_fov = args["well_fov"]
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0014_T1"
    well_fov = "G2-2"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[3]:


input_sqlite_file = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/{well_fov}.duckdb"
).resolve(strict=True)
destination_sc_parquet_file = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/sc_profiles_{well_fov}.parquet"
).resolve()
destination_organoid_parquet_file = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/organoid_profiles_{well_fov}.parquet"
).resolve()
destination_sc_parquet_file.parent.mkdir(parents=True, exist_ok=True)
dest_datatype = "parquet"


# In[4]:


# show the tables
with duckdb.connect(input_sqlite_file) as con:
    tables = con.execute("SHOW TABLES").fetchdf()
    print(tables)
    nuclei_table = con.sql("SELECT * FROM Nuclei").df()
    cells_table = con.sql("SELECT * FROM Cell").df()
    cytoplasm_table = con.sql("SELECT * FROM Cytoplasm").df()
    organoid_table = con.sql("SELECT * FROM Organoid").df()


# In[5]:


nuclei_id_set = set(nuclei_table["object_id"].to_list())
cells_id_set = set(cells_table["object_id"].to_list())
cytoplasm_id_set = set(cytoplasm_table["object_id"].to_list())
# find the intersection of the three sets
intersection_set = nuclei_id_set.intersection(cells_id_set, cytoplasm_id_set)
# keep only the rows in the three tables that are in the intersection set
nuclei_table = nuclei_table[nuclei_table["object_id"].isin(intersection_set)]
cells_table = cells_table[cells_table["object_id"].isin(intersection_set)]
cytoplasm_table = cytoplasm_table[cytoplasm_table["object_id"].isin(intersection_set)]


# In[6]:


# connect to DuckDB and register the tables
with duckdb.connect() as con:
    con.register("nuclei", nuclei_table)
    con.register("cells", cells_table)
    con.register("cytoplasm", cytoplasm_table)
    # Merge them with SQL
    merged_df = con.execute("""
        SELECT *
        FROM nuclei
        LEFT JOIN cells USING (object_id)
        LEFT JOIN cytoplasm USING (object_id)
    """).df()


# In[7]:


# save the organoid data as parquet
print(f"Final organoid data shape: {merged_df.shape}")
organoid_table.to_parquet(destination_organoid_parquet_file, index=False)
organoid_table.head()


# In[8]:


# drop columns that end with _x or _y lowercase
merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith(("_x", "_y"))]


# In[9]:


print(f"Final merged single cell dataframe shape: {merged_df.shape}")
# save the sc data as parquet
merged_df.to_parquet(destination_sc_parquet_file, index=False)
merged_df.head()
