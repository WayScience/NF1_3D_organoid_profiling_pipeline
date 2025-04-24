#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# In[2]:


well_fov = "C4-2"
stats_path = pathlib.Path("../../2.cellprofiling/results/run_stats/").resolve(
    strict=True
)
output_path = pathlib.Path(f"../results/converted_profiles/{well_fov}").resolve()
output_path.mkdir(parents=True, exist_ok=True)

stats_files = list(stats_path.glob("*.parquet"))
stats_files.sort()
stats_files


# In[3]:


df = pd.concat(
    [pd.read_parquet(stats_file) for stats_file in stats_files],
    ignore_index=True,
)
df.to_parquet(output_path / f"{well_fov}_cell_stats.parquet", index=False)
df["feature_type_and_gpu"] = (
    df["feature_type"].astype(str) + "_" + df["gpu"].astype(str)
)
df["feature_type_and_gpu"] = df["feature_type_and_gpu"].str.replace("None", "CPU")
df["feature_type_and_gpu"] = df["feature_type_and_gpu"].str.replace("True", "GPU")
df["time_taken_minutes"] = df["time_taken"] / 60
df["mem_usage_GB"] = df["mem_usage"] / (1024)
df.head(10)


# ## Preliminary plots - will finalize in R later

# In[4]:


# plot the memory and time for each feature type

sns.barplot(
    data=df,
    x="feature_type_and_gpu",
    y="time_taken_minutes",
    hue="feature_type",
    palette="Set2",
)
plt.title("Time taken for each feature type")
plt.xlabel("Feature Type")
plt.ylabel("Time (min)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
# move the legend outside the plot
plt.legend(title="Feature Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

sns.barplot(
    data=df,
    x="feature_type_and_gpu",
    y="mem_usage_GB",
    hue="feature_type",
    palette="Set2",
)
plt.title("Memory used for each feature type")
plt.xlabel("Feature Type")
plt.ylabel("Memory (GB)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.legend(title="Feature Type", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.show()
