list_of_packages <- c("ggplot2", "dplyr", "tidyr", "arrow", "ggbreak")
for (package in list_of_packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(
                package,
                character.only = TRUE,
                quietly = TRUE,
                warn.conflicts = FALSE
            )
        )
    )
}
# Get the current working directory and find Git root
find_git_root <- function() {
    # Get current working directory
    cwd <- getwd()

    # Check if current directory has .git
    if (dir.exists(file.path(cwd, ".git"))) {
        return(cwd)
    }

    # If not, search parent directories
    current_path <- cwd
    while (dirname(current_path) != current_path) {  # While not at root
        parent_path <- dirname(current_path)
        if (dir.exists(file.path(parent_path, ".git"))) {
            return(parent_path)
        }
        current_path <- parent_path
    }

    # If no Git root found, stop with error
    stop("No Git root directory found.")
}

# Find the Git root directory
root_dir <- find_git_root()
cat("Git root directory:", root_dir, "\n")

bandicoot_path <- normalizePath(
    file.path("~", "mnt", "bandicoot","NF1_organoid_data"),
    mustWork = FALSE
)
if (!dir.exists(bandicoot_path)) {
    profile_base_dir <- file.path(
        root_dir
    )
} else {
    profile_base_dir <- file.path(
        bandicoot_path
    )
}
profile_base_dir

# get the profiling stats and load into a dataframe
profiling_path <- file.path(
    profile_base_dir,
    "data/all_patient_profiles/all_patient_featurization_stats.parquet"
)
profiling_stats_df <- arrow::read_parquet(profiling_path)
head(profiling_stats_df)
figures_path <- file.path(
    root_dir,
    "4.processing_image_based_profiles/figures/"
)
if (!dir.exists(figures_path)) {
    dir.create(figures_path, recursive = TRUE)
}

width <- 12
height <- 6
options(repr.plot.width = width, repr.plot.height = height)
# plot the time taken for each sub-image set
# where a sub-image set is a channel-compartment-image-set combination
time_plot <- (
    ggplot(
        profiling_stats_df,
        aes(
            x = feature_type,
            y = time_taken_minutes,
            fill = feature_type
        )
    )
    + geom_boxplot(outlier.shape = NA)
    + geom_jitter(
        aes(color = feature_type),
        position = position_jitterdodge(dodge.width = 0.1, jitter.width = 0.6),
        alpha = 0.1
    )
    + theme_bw()
    + theme(
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        legend.position = "none",
    )
    + labs(
        x = "Feature Type",
        y = "Time Taken (minutes)",
    )
)
# add a break in the y-axis to highlight the outliers
time_plot <- time_plot + scale_y_break(c(15, 16),scales = 0.3, space = 0.05)
# save the plot to the figures directory
ggsave(
    filename = file.path(figures_path, "profiling_time_per_feature_type.png"),
    plot = time_plot,
    width = width,
    height = height,
    dpi = 300
)
time_plot

# plot the memory usage for each sub-image set
# where a sub-image set is a channel-compartment-image-set combination
mem_plot <- (
    ggplot(
        profiling_stats_df,
        aes(
            x = feature_type,
            y = mem_usage_GB,
            fill = feature_type
        )
    )
    + geom_boxplot(outlier.shape = NA)
    + geom_jitter(
        aes(color = feature_type),
        position = position_jitterdodge(dodge.width = 0.9, jitter.width = 0.5),
        alpha = 0.1
    )
    + theme_bw()
    + theme(
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        legend.position = "none",
    )
    + labs(
        x = "Feature Type",
        y = "Memory Usage (GB)",
    )
)
ggsave(
    filename = file.path(figures_path, "profiling_memory_per_feature_type.png"),
    plot = mem_plot,
    width = width,
    height = height,
    dpi = 300
)
mem_plot
