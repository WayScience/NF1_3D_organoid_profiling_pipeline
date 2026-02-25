list_of_packages <- c("ggplot2", "dplyr", "tidyr", "circlize")
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

plot_theme <- theme_bw() + theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 16),
        strip.text = element_text(size = 14)
    )

figures_2D_path <- file.path("../figures/microscope_comparisons_2D")
figures_3D_path <- file.path("../figures/microscope_comparisons_3D")
if (!dir.exists(figures_2D_path)) {
  dir.create(figures_2D_path, recursive = TRUE)
}
if (!dir.exists(figures_3D_path)) {
  dir.create(figures_3D_path, recursive = TRUE)
}

analysis_2D_file_path = file.path(
    "../results/raw_image_quality_metrics/merged_results_2D.parquet"
)
analysis_3D_file_path = file.path(
    "../results/raw_image_quality_metrics/merged_results_3D.parquet"
)
raw_image_2D_quality_metrics <- arrow::read_parquet(analysis_2D_file_path)
raw_image_3D_quality_metrics <- arrow::read_parquet(analysis_3D_file_path)

raw_image_2D_quality_metrics <- as.data.frame(raw_image_2D_quality_metrics)
raw_image_3D_quality_metrics <- as.data.frame(raw_image_3D_quality_metrics)
# make the compartment a factor
raw_image_2D_quality_metrics$compartment <- factor(
    raw_image_2D_quality_metrics$compartment,
    levels = c("organoid", "nuclei", "cell")
)
raw_image_3D_quality_metrics$compartment <- factor(
    raw_image_3D_quality_metrics$compartment,
    levels = c("organoid", "nuclei", "cell")
)
# if CQ1 in patient then set microscope to CQ1
raw_image_2D_quality_metrics$microscope <- ifelse(
    grepl("CQ1", raw_image_2D_quality_metrics$patient),
    "CQ1", # set to CQ1 if CQ1 in patient
    "Echo" # else set to Echo
)
raw_image_3D_quality_metrics$microscope <- ifelse(
    grepl("CQ1", raw_image_3D_quality_metrics$patient),
    "CQ1", # set to CQ1 if CQ1 in patient
    "Echo" # else set to Echo
)
head(raw_image_2D_quality_metrics)
head(raw_image_3D_quality_metrics)

height <- 6
width <- 8
options(repr.plot.width = width, repr.plot.height = height)
s_n_ratio_plot <- (
    ggplot(
        data = raw_image_3D_quality_metrics,
        aes(
            x = channel,
            y = signal_to_noise_ratio,
            fill = microscope
    )
    )
    + geom_boxplot(
        # add jitter and transparency to see all points
        outlier.size = 0.5,
        position = position_dodge(width = 0.75)
    )
    + geom_jitter(
        aes(
            color = microscope
        ),
        position = position_jitterdodge(
            jitter.width = 0.2,
            dodge.width = 0.75
        ),
        alpha = 0.3,
        size = 0.5
    )
    + labs(
        x = "Channel",
        y = "Signal to Noise Ratio"
    )
    + plot_theme
)
ggsave(
    filename = file.path(
        figures_3D_path,
        "signal_to_noise_ratio_by_microscope_and_channel_3D.png"
    ),
    plot = s_n_ratio_plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600
)
s_n_ratio_plot

height <- 6
width <- 8
options(repr.plot.width = width, repr.plot.height = height)
michelson_contrast_plot <- (
    ggplot(
        data = raw_image_3D_quality_metrics,
        aes(
            x = channel,
            y = michelson_contrast,
            fill = microscope
    )
    )
    + geom_boxplot(
        # add jitter and transparency to see all points
        outlier.size = 0.5,
        position = position_dodge(width = 0.75)
    )
    + geom_jitter(
        aes(
            color = microscope
        ),
        position = position_jitterdodge(
            jitter.width = 0.2,
            dodge.width = 0.75
        ),
        alpha = 0.3,
        size = 0.5
    )
    + labs(
        x = "Channel",
        y = "Michelson Contrast"
    )
    + ylim(0,1)
    + plot_theme

)
ggsave(
    filename = file.path(
        figures_3D_path,
        "michelson_contrast_by_microscope_and_channel_3D.png"
    ),
    plot = michelson_contrast_plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600
)
michelson_contrast_plot

height <- 6
width <- 8
options(repr.plot.width = width, repr.plot.height = height)
RMS_contrast_plot <- (
    ggplot(
        data = raw_image_3D_quality_metrics,
        aes(
            x = channel,
            y = RMS_contrast,
            fill = microscope
    )
    )
    + geom_boxplot(
        # add jitter and transparency to see all points
        outlier.size = 0.5,
        position = position_dodge(width = 0.75)
    )
    + geom_jitter(
        aes(
            color = microscope
        ),
        position = position_jitterdodge(
            jitter.width = 0.2,
            dodge.width = 0.75
        ),
        alpha = 0.3,
        size = 0.5
    )
    + labs(
        x = "Channel",
        y = "RMS Contrast"
    )
    + plot_theme
)
ggsave(
    filename = file.path(
        figures_3D_path,
        "RMS_contrast_by_microscope_and_channel_3D.png"
    ),
    plot = RMS_contrast_plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600
)
RMS_contrast_plot

# normalize the per channel per microscope per well fov the z slice values
# to be between 0 and 1 for each channel and microscope combination
raw_image_2D_quality_metrics <- raw_image_2D_quality_metrics %>%
    group_by(microscope, channel, well_fov) %>%
    mutate(
        z_slice_normalized = (z_slice - min(z_slice)) / (max(z_slice) - min(z_slice))
    )
head(raw_image_2D_quality_metrics)

height <- 6
width <- 8
options(repr.plot.width = width, repr.plot.height = height)
s_n_ratio_plot <- (
    ggplot(
        data = raw_image_2D_quality_metrics,
        aes(
            x = z_slice_normalized,
            y = signal_to_noise_ratio,
            color = microscope
    )
    )
    + geom_point(
        alpha = 0.1,
        size = 0.2
    )
    + labs(
        x = "Z Slice",
        y = "Signal to Noise Ratio"
    )
    + facet_wrap(. ~ channel, scales = "free_y", nrow = 2)
    + guides(
        color = guide_legend(
            title = "Microscope", override.aes = list(
                size = 2,
                alpha = 1
            )
        ),
        text = element_text(size = 16)
    )
    + theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 16),
        strip.text = element_text(size = 14)
    )
    + plot_theme
)
ggsave(
    filename = file.path(
        figures_2D_path,
        "signal_to_noise_ratio_by_microscope_and_channel_2D.png"
    ),
    plot = s_n_ratio_plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600
)
s_n_ratio_plot

height <- 6
width <- 8
options(repr.plot.width = width, repr.plot.height = height)
michelson_contrast_plot <- (
    ggplot(
        data = raw_image_2D_quality_metrics,
        aes(
            x = z_slice_normalized,
            y = michelson_contrast,
            color = microscope
    )
    )
    + geom_point(
        alpha = 0.1,
        size = 0.2
    )
    + labs(
        x = "Channel",
        y = "Michelson Contrast"
    )
    + ylim(0,1)
    + facet_wrap(. ~ channel, scales = "free_y", nrow = 2)
    + guides(
        color = guide_legend(
            title = "Microscope", override.aes = list(
                size = 2,
                alpha = 1
            )
        )
    )
    + plot_theme

)
ggsave(
    filename = file.path(
        figures_2D_path,
        "michelson_contrast_by_microscope_and_channel_2D.png"
    ),
    plot = michelson_contrast_plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600
)
michelson_contrast_plot

height <- 6
width <- 8
options(repr.plot.width = width, repr.plot.height = height)
RMS_contrast_plot <- (
    ggplot(
        data = raw_image_2D_quality_metrics,
        aes(
            x = z_slice_normalized,
            y = RMS_contrast,
            color = microscope
    )
    )
    + geom_point(
        alpha = 0.1,
        size = 0.2
    )
    + labs(
        x = "Z Slice Normalized",
        y = "RMS Contrast"
    )
    + facet_wrap(. ~ channel, scales = "free_y", nrow = 2)
    + guides(
        color = guide_legend(
            title = "Microscope", override.aes = list(
                size = 2,
                alpha = 1
            )
        )
    )
    + plot_theme

)
ggsave(
    filename = file.path(
        figures_2D_path,
        "RMS_contrast_by_microscope_and_channel_2D.png"
    ),
    plot = RMS_contrast_plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600
)
RMS_contrast_plot
