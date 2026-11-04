list_of_packages <- c("ggplot2", "dplyr", "arrow", "ggsignif")
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
git_root <- system("git rev-parse --show-toplevel", intern = TRUE)
plot_theme <- theme_bw() + theme(
        axis.text = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_text(size = 16),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 16),
        strip.text = element_text(size = 14),
        plot.title = element_text(size = 18, hjust = 0.5)
    )

figures_path <- file.path(git_root, "5.technical_analysis/figures/zslice_segmentation_comparison")
if (!dir.exists(figures_path)) {
  dir.create(figures_path, recursive = TRUE)
}

processed_data_dir <- file.path(git_root, "5.technical_analysis/processed_data")
iou_results_path <- file.path(processed_data_dir, "zslice_iou_results.parquet")
per_compartment_volume_path <- file.path(processed_data_dir, "zslice_per_compartment_volume.parquet")


iou_results_df <- as.data.frame(arrow::read_parquet(iou_results_path))
per_compartment_volume_df <- as.data.frame(arrow::read_parquet(per_compartment_volume_path))

head(iou_results_df)
head(per_compartment_volume_df)


# match the compartment ordering used in the Python notebook
compartment_levels <- c("organoid", "nuclei", "cytoplasm", "cell")
iou_results_df$compartment <- factor(iou_results_df$compartment, levels = compartment_levels)
per_compartment_volume_df$compartment <- factor(
    per_compartment_volume_df$compartment, levels = compartment_levels
)

# fixed, colorblind-friendly (Okabe-Ito) qualitative palette for compartment --
# these are unordered categories, so a perceptual/sequential palette like viridis
# isn't the right tool here
compartment_palette <- c(
    "organoid" = "#E69F00",
    "nuclei" = "#56B4E9",
    "cytoplasm" = "#009E73",
    "cell" = "#CC79A7"
)

# treat 0.1um (the finest step tested) as the reference, and only keep the 3
# z-step comparisons that compare a coarser step against it -- drop the 3
# coarser-vs-coarser comparisons (e.g. 1um vs 0.5um), which aren't of interest here
z_comparison_levels <- c("1um vs 0.1um", "0.5um vs 0.1um", "0.2um vs 0.1um")
iou_results_df <- iou_results_df %>% filter(z_comparison %in% z_comparison_levels)
iou_results_df$z_comparison <- factor(iou_results_df$z_comparison, levels = z_comparison_levels)

# every pairwise combination among the remaining 3 comparisons, for the
# significance brackets below
z_comparison_pairs <- combn(z_comparison_levels, 2, simplify = FALSE)

# qualitative (ColorBrewer "Dark2") palette for the 3 remaining z-step comparisons
z_comparison_palette <- c(
    "1um vs 0.1um" = "#1B9E77",
    "0.5um vs 0.1um" = "#D95F02",
    "0.2um vs 0.1um" = "#7570B3"
)

z_slice_step_levels <- c("1um", "0.5um", "0.2um", "0.1um")
per_compartment_volume_df$z_slice_step <- factor(
    per_compartment_volume_df$z_slice_step, levels = z_slice_step_levels
)

# again treat 0.1um as the reference step: only compare each coarser step against
# 0.1um, not against each other
z_slice_step_pairs <- lapply(
    setdiff(z_slice_step_levels, "0.1um"), function(step) c(step, "0.1um")
)

head(iou_results_df)
head(per_compartment_volume_df)


width <- 18
height <- 10
options(repr.plot.width = width, repr.plot.height = height)

# one boxplot per z-step comparison (not dodged), faceted by compartment x patient, so
# pairwise significance brackets have unambiguous x-positions to anchor to.
# 0.1um (the finest step tested) is treated as the reference, so only the 3
# comparisons against it are kept (see z_comparison_levels above); all 3 pairwise
# combinations among those are tested (Wilcoxon rank-sum, computed per facet by
# geom_signif itself) and labeled ns / * / ** / *** -- not just the significant ones --
# so absence of a difference is visible too, not only presence of one.
iou_plot <- (
    ggplot(
        data = iou_results_df,
        aes(x = z_comparison, y = IOU, fill = z_comparison)
    )
    + geom_boxplot(outlier.size = 0.5)
    + geom_signif(
        comparisons = z_comparison_pairs,
        test = "wilcox.test",
        map_signif_level = TRUE,
        step_increase = 0.08,
        tip_length = 0.01,
        textsize = 3
    )
    + scale_fill_manual(values = z_comparison_palette)
    + facet_grid(rows = vars(patient), cols = vars(compartment))
    + scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0.05, 0.4)))
    + labs(
        x = "Z-Step Comparison (vs. 0.1um reference)",
        y = "Intersection over Union (IOU)",
        title = "IOU of Segmentations: Each Z-Step Size vs. the 0.1um Reference",
        fill = "Z-Step Comparison"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_path, "iou_by_zstep_comparison.png"),
    plot = iou_plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600
)
iou_plot


# same restructuring as iou_plot: one box per z-step comparison (vs. the 0.1um
# reference), faceted by compartment x patient, with the 3 pairwise Wilcoxon
# comparisons among them annotated
volume_ratio_plot <- (
    ggplot(
        data = iou_results_df,
        aes(x = z_comparison, y = volume_ratio, fill = z_comparison)
    )
    + geom_boxplot(outlier.size = 0.5)
    + geom_signif(
        comparisons = z_comparison_pairs,
        test = "wilcox.test",
        map_signif_level = TRUE,
        step_increase = 0.08,
        tip_length = 0.01,
        textsize = 3
    )
    + scale_fill_manual(values = z_comparison_palette)
    + facet_grid(rows = vars(patient), cols = vars(compartment))
    + scale_y_continuous(expand = expansion(mult = c(0.05, 0.4)))
    + labs(
        x = "Z-Step Comparison (vs. 0.1um reference)",
        y = "Volume Ratio",
        title = "Volume Ratio: Each Z-Step Size vs. the 0.1um Reference",
        fill = "Z-Step Comparison"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_path, "volume_ratio_by_zstep_comparison.png"),
    plot = volume_ratio_plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600
)
volume_ratio_plot


# the Python notebook saves the normalized table directly (so both notebooks agree on
# the normalization); load it if present, otherwise compute + persist it here
per_compartment_volume_normalized_path <- file.path(
    processed_data_dir, "zslice_per_compartment_volume_normalized.parquet"
)
if (file.exists(per_compartment_volume_normalized_path)) {
    per_compartment_volume_df <- as.data.frame(
        arrow::read_parquet(per_compartment_volume_normalized_path)
    )
    per_compartment_volume_df$compartment <- factor(
        per_compartment_volume_df$compartment, levels = compartment_levels
    )
    per_compartment_volume_df$z_slice_step <- factor(
        per_compartment_volume_df$z_slice_step, levels = z_slice_step_levels
    )
} else {
    # min-max normalize per patient + compartment (same definition as the Python
    # notebook), since absolute volume scale differs across patients
    per_compartment_volume_df <- per_compartment_volume_df %>%
        group_by(patient, compartment) %>%
        mutate(
            normalized_volume = (volume - min(volume)) / (max(volume) - min(volume))
        ) %>%
        ungroup()
    arrow::write_parquet(per_compartment_volume_df, per_compartment_volume_normalized_path)
}


# same restructuring as iou_plot/volume_ratio_plot: one box per z-slice-step,
# faceted by compartment x patient; 0.1um is the reference step, so only the 3
# pairwise Wilcoxon comparisons against it are annotated (see z_slice_step_pairs above)
normalized_volume_plot <- (
    ggplot(
        data = per_compartment_volume_df,
        aes(x = z_slice_step, y = normalized_volume, fill = z_slice_step)
    )
    + geom_boxplot(outlier.size = 0.5)
    + geom_signif(
        comparisons = z_slice_step_pairs,
        test = "wilcox.test",
        map_signif_level = TRUE,
        step_increase = 0.08,
        tip_length = 0.01,
        textsize = 3
    )
    + facet_grid(rows = vars(patient), cols = vars(compartment))
    + scale_y_continuous(expand = expansion(mult = c(0.05, 0.4)))
    + labs(
        x = "Z-Step Size (vs. 0.1um reference)",
        y = "Normalized Volume",
        title = "Volume of Segmentations: Each Z-Step Size vs. the 0.1um Reference",
        fill = "Z-Step Size"
    )
    + plot_theme
)
ggsave(
    filename = file.path(figures_path, "normalized_volume_by_zstep.png"),
    plot = normalized_volume_plot,
    width = width,
    height = height,
    units = "in",
    dpi = 600
)
normalized_volume_plot

