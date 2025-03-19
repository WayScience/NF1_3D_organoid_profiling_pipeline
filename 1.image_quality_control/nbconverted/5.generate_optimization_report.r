# Load necessary packages
suppressPackageStartupMessages({
    library(arrow)
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(colorspace)
})

# Set up directory for plots
figures_dir <- "../qc_figures"

# Create the directory if it doesn't exist
if (!dir.exists(figures_dir)) {
    dir.create(figures_dir)
}

# Load the dataframe from a path
qc_results_df <- read_parquet("../qc_results/NF0017_qc_results_optimization.parquet")

# Check for any NaNs in the columns starting with Metadata_
metadata_cols <- grep("^Metadata_", colnames(qc_results_df), value = TRUE)
na_counts <- sapply(qc_results_df[metadata_cols], function(x) sum(is.na(x)))

# Print the count of NaNs for each Metadata_ column
na_counts

# Look at the dimensions and head of the dataframe
dim(qc_results_df)
head(qc_results_df)

qc_type_cols <- grep("Blurry|Saturated", colnames(qc_results_df), value = TRUE)

# Assuming qc_results_df is your dataframe and condition_cols is a vector of column names
melted_qc_df <- qc_results_df %>%
  pivot_longer(
    cols = all_of(qc_type_cols),
    names_to = c("QC_type", "Channel"),
    names_sep = "_",
    values_to = "Failed"
  ) %>%
  mutate(
    Condition_combo = paste0("(", AGP_conditions, ")-(", Mito_conditions, ")")
  )

dim(melted_qc_df)
head(melted_qc_df)

# Group by Well, Site, and Condition_combo and check if there is at least one failing z-slice
failed_condition_combos <- melted_qc_df %>%
  group_by(Metadata_Well, Metadata_Site, Condition_combo) %>%
  summarise(At_least_one_fail = any(Failed == TRUE), .groups = 'drop') %>%
  filter(At_least_one_fail == TRUE) %>%
  group_by(Metadata_Well, Metadata_Site) %>%
  summarise(Failed_condition_combos_count = n(), .groups = 'drop')

# Print the result
print(failed_condition_combos)

# Count unique AGP_condition and Mito_condition combinations
unique_combos_count <- melted_qc_df %>%
  distinct(Condition_combo) %>%
  summarise(Unique_combos = n())

# Print the number of unique combinations
print(unique_combos_count)

# Assuming melted_qc is your dataframe
count_failed_qc <- melted_qc_df %>%
  filter(Failed == TRUE) %>%
  group_by(Condition_combo, QC_type, Channel) %>%
  summarise(Failed_count = n(), .groups = 'drop')

dim(count_failed_qc)
count_failed_qc

# Set width and height
width = 12
height = 8
options(repr.plot.width = width, repr.plot.height = height)

# Plot the bar chart of counts of failed z-slices for each condition and channel
count_zslices_channel <- ggplot(count_failed_qc, aes(x = Channel, y = Failed_count, fill = Condition_combo)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_fill_brewer(palette = "Set2") +
    facet_grid(~QC_type) +
    labs(
        title = "Counts of z-slices that failed at least one condition",
        x = "Channel",
        y = "Count",
        fill = "Imaging conditions:\n(AGP channel)-(Mito channel)\n(LED power-Exposure time)"
    ) +
    theme_bw() +
    theme(
        plot.title = element_text(size = 20),
        axis.title = element_text(size = 16),
        axis.text = element_text(size = 14),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 14)
    )

# Show plot
print(count_zslices_channel)

# Save plot
ggsave(file.path(figures_dir, "optimization_count_zslices_channel.png"), plot = count_zslices_channel, width = width, height = height, dpi = 500)


# Group by Metadata_Zslice, Condition, and Channel, then summarize the number of failed zslices
failed_zslices_per_metadata <- melted_qc_df %>%
    group_by(Metadata_Zslice, QC_type, Channel, Condition_combo) %>%
    summarize(Failed_Count = sum(Failed == TRUE, na.rm = TRUE)) %>%  # Explicitly count TRUE values
    ungroup()

# Show dimension and head of the resulting dataframe
dim(failed_zslices_per_metadata)
head(failed_zslices_per_metadata)

# Calculate the maximum Failed_Count across the entire dataset
max_failed_count <- max(failed_zslices_per_metadata$Failed_Count, na.rm = TRUE)

# Set width and height
width = 22
height = 10
options(repr.plot.width = width, repr.plot.height = height)

# Create the bar plot with consistent y-axis limits
bar_plot <- ggplot(failed_zslices_per_metadata, aes(x = Metadata_Zslice, y = Failed_Count, fill = Condition_combo)) +
    geom_bar(stat = "identity", position = "dodge", linewidth = 2) +
    facet_grid(Channel ~ QC_type) +
    scale_fill_brewer(palette = "Set2") +
    labs(
        title = "Count of failed z-slices per channel across conditions",
        x = "Z-slice",
        y = "Count",
        fill = "Imaging conditions:\n(AGP channel)-(Mito channel)\n(LED power-Exposure time)"
    ) +
    theme_bw() +
    theme(
        plot.title = element_text(size = 24),
        axis.title = element_text(size = 20),
        axis.text.y = element_text(size = 18),
        axis.text.x = element_text(size = 18, angle = 90, hjust = 1),
        legend.title = element_text(size = 20),
        legend.text = element_text(size = 18),
        strip.text = element_text(size = 20)
    ) +
    ylim(0, max_failed_count)
    
# Show plot
print(bar_plot)

# Save plot
ggsave(file.path(figures_dir, "optimization_failed_zslice_count_channel_and_condition.png"), plot = bar_plot, width = width, height = height, dpi = 500)

# Step 1: Remove duplicates for z-slices per organoid (Plate, Well, Site, Zslice, Condition combo)
unique_zslices <- melted_qc_df %>%
    distinct(Metadata_Plate, Metadata_Well, Metadata_Site, Metadata_Zslice, Condition_combo) %>%
    mutate(Numeric_Zslice = as.numeric(gsub("ZS", "", Metadata_Zslice)))

# Step 2: Normalize the z-slices per organoid
normalized_zslices <- unique_zslices %>%
    group_by(Metadata_Plate, Metadata_Well, Metadata_Site) %>%
    mutate(
        Normalized_Zslice = (Numeric_Zslice - min(Numeric_Zslice)) / 
                            (max(Numeric_Zslice) - min(Numeric_Zslice))
    ) %>%
    ungroup()

# Step 3: Join the normalized z-slices back to the original dataframe
norm_melted_qc_df <- melted_qc_df %>%
    left_join(normalized_zslices, by = c("Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_Zslice", "Condition_combo"))

# Step 4: Inspect the result
dim(norm_melted_qc_df)
head(norm_melted_qc_df)

# Group by Metadata_Zslice, QC_type, Condition_combo, and Channel, then summarize the number of failed zslices
norm_failed_zslices_per_metadata <- norm_melted_qc_df %>%
    group_by(Normalized_Zslice, QC_type, Condition_combo, Channel) %>%
    summarize(Failed_Count = sum(Failed == TRUE, na.rm = TRUE)) %>%  # Explicitly count TRUE values
    ungroup()

# Show dimension and head of the resulting dataframe
dim(norm_failed_zslices_per_metadata)
head(norm_failed_zslices_per_metadata)

# Set width and height
width <- 15
height <- 10
options(repr.plot.width = width, repr.plot.height = height)

# Create the bar plot using Failed_Count
histogram_plot <- ggplot(norm_failed_zslices_per_metadata, aes(x = Normalized_Zslice, y = Failed_Count, fill = Condition_combo)) +
    geom_bar(stat = "identity", alpha = 0.5, position = "stack", width = 0.03) +
    facet_grid(Channel ~ QC_type) +
    scale_fill_brewer(palette = "Set2") +
    labs(
        title = "Count of failed normalized z-slices per channel across conditions",
        x = "Normalized z-slice",
        y = "Failed count",
        fill = "Imaging conditions:\n(AGP channel)-(Mito channel)\n(LED power-Exposure time)"
    ) +
    theme_bw() +
    theme(
        plot.title = element_text(size = 24),
        axis.title = element_text(size = 20),
        axis.text.y = element_text(size = 18),
        axis.text.x = element_text(size = 18, angle = 90, hjust = 1),
        legend.title = element_text(size = 19),
        legend.text = element_text(size = 18),
        strip.text = element_text(size = 20)
    )

# Show plot
print(histogram_plot)

# Save plot
ggsave(file.path(figures_dir, "optimization_failed_norm_zslice_count_channel_and_condition.png"), plot = histogram_plot, width = width, height = height, dpi = 500)

