# Treatment color palette, vendored from WayScience/NF1_organoid_profile_analysis.
# Source: https://github.com/WayScience/NF1_organoid_profile_analysis/blob/main/utils/r_plot_themes.r
# Only the drug/treatment -> color mapping is vendored here (not the full
# theme file, which also covers unrelated plots: channels, viability, UpSet, etc.)
#
# Hue = MOA family (similar mechanisms share a hue range); Lightness/Saturation = dose
custom_treatment_palette <- c(
  "DMSO" = "#A6A6A6", # Control
  "Staurosporine" = "#3F468C", # Broad-spectrum kinase inhibitor
  "Fimepinostat" = "#3DCCA8", # HDAC/PI3K inhibitor (dual)
  "Copanlisib" = "#3DCACC", # PI3K inhibitor
  "Everolimus" = "#3DA4CC", # mTOR inhibitor
  "Rapamycin" = "#3D7DCC", # mTOR inhibitor
  "Sapanisertib" = "#3D57CC", # mTOR inhibitor
  "Vistusertib" = "#493DCC", # mTOR inhibitor
  "Panobinostat" = "#CC8029", # HDAC inhibitor
  "ARV-825" = "#CCAB29", # BRD4 inhibitor
  "Imatinib" = "#6047CC", # BCR-ABL/KIT inhibitor
  "Nilotinib" = "#8347CC", # BCR-ABL/KIT inhibitor
  "Cabozantinib" = "#A647CC", # Multi-kinase inhibitor
  "Linsitinib" = "#CA47CC", # IGF-1R inhibitor
  "Binimetinib" = "#D92B7F", # MEK inhibitor
  "Mirdametinib" = "#D92B51", # MEK inhibitor
  "Trametinib" = "#D9342B", # MEK inhibitor
  "Selumetinib" = "#D9622B", # MEK inhibitor
  "Onalespib" = "#6CA642", # HSP90 inhibitor
  "Digoxin" = "#BF3078", # Na/K-ATPase inhibitor
  "Ketotifen" = "#238C83", # Antihistamine
  "Trabectedin" = "#388C5B" # DNA-binding agent
)

# Patient/tumor-barcode -> tumor type, and tumor type -> color. NF0030_T1
# (myopericytoma) and NF0040_T1 (schwannoma) are grouped as "Other" because
# they are not NF1 nerve-sheath tumors.
tumor_type_lookup <- c(
  "NF0014_T1" = "cNF",
  "NF0014_T2" = "pNF",
  "NF0016_T1" = "pNF",
  "NF0018_T6" = "cNF",
  "NF0021_T1" = "cNF",
  "NF0030_T1" = "Other",
  "NF0035_T1" = "cNF",
  "NF0037_T1" = "cNF",
  "NF0040_T1" = "Other",
  "NF0055_T1" = "pNF",
  "SARCO219_T2" = "MPNST",
  "SARCO361_T1" = "MPNST"
)

tumor_type_palette <- c(
  "cNF" = "#1B9E77",
  "pNF" = "#D95F02",
  "MPNST" = "#7570B3",
  "Other" = "#999999"
)
