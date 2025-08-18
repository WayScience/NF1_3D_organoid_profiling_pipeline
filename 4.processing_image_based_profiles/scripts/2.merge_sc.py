#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import pathlib
import sys

import pandas as pd
from cytotable import convert, presets

sys.path.append("../../../utils")
import uuid

import duckdb
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

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


# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID to process, e.g. 'P01'",
    )
    argparser.add_argument(
        "--well_fov",
        type=str,
        required=True,
        help="Well and field of view to process, e.g. 'A01_1'",
    )
    args = argparser.parse_args()
    patient = args.patient
    well_fov = args.well_fov
else:
    patient = "NF0014"
    well_fov = "C4-2"


# In[ ]:


input_sqlite_file = pathlib.Path(
    f"{root_dir}/data/{patient}/converted_profiles/{well_fov}/{well_fov}.duckdb"
).resolve(strict=True)
destination_sc_parquet_file = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/{well_fov}/sc_profiles_{well_fov}.parquet"
).resolve()
destination_organoid_parquet_file = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/{well_fov}/organoid_profiles_{well_fov}.parquet"
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
    con.register("df1", nuclei_table)
    con.register("df2", cells_table)
    con.register("df3", cytoplasm_table)
    # Merge them with SQL
    merged_df = con.execute("""
        SELECT *
        FROM df1
        LEFT JOIN df2 USING (object_id)
        LEFT JOIN df3 USING (object_id)
    """).df()


# In[7]:


# save the organoid data as parquet
print(f"Final organoid data shape: {merged_df.shape}")
organoid_table.to_parquet(destination_organoid_parquet_file, index=False)
organoid_table.head()


# In[ ]:


print(f"Final merged single cell dataframe shape: {merged_df.shape}")
# save the sc data as parquet
merged_df.to_parquet(destination_sc_parquet_file, index=False)
merged_df.head()

