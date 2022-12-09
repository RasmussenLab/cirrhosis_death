######Olink QC ProDoc################

#title: "Bridge Normalization & QC for Prodoc Data"
#date: "04/11/2022"


#loading packages
library(tidyverse)
library(OlinkAnalyze)
library(readxl)


##loading ProDoc file with NPX values
ProD <- OlinkAnalyze::read_NPX(filename = 'H:/From_SUND/Projects/ProDoc/BX0418_Data.xlsx')

#Loading CBMRIDS
prod_cbmrid <- read_excel('H:/From_SUND/Projects/ProDoc/id.xlsx')

#Merging CBMRIDs

ProD_ID <- left_join(ProD, prod_cbmrid, by=c('SampleID'))

ProD_ID <- ProD_ID %>%
  rename('BXID' = 'SampleID') %>%
  rename('SampleID' = 'CBMRID')

## LOAD microbliver BATCH 
## LOAD MicrobLiver BATCH 1 (N = 1,808)
MLB1 <- OlinkAnalyze::read_NPX(filename = "J:/CBMR/SUN-CBMR-Hansen-Group/Projects/MicrobLiver/Results/Olink/Raw_data_reports/20202249_Juel_NPX.xlsx")

#Adding project number
## ADD PROJECT NUMBER to both projects
MLB1_df1 <- MLB1 %>% dplyr::mutate(Project = '20202249')

ProD_df1 <- ProD_ID %>% dplyr::mutate(Project = 'BX0418')

## RENAME BRIDGE MLB1_INF_df1 (CONSISTENCY)
MLB1_df1 <- MLB1_df1 %>%
  dplyr::mutate(SampleID = gsub("BRIDGE0", "Bridge_", SampleID)) %>%
  dplyr::mutate(SampleID = gsub("BRIDGE", "Bridge_", SampleID))


# ProD_df1 <- ProD_df1 %>%
#   dplyr::mutate(SampleID = gsub("Bridge0", "Bridge_", SampleID)) %>%
#   dplyr::mutate(SampleID = gsub("Bridge", "Bridge_", SampleID))

## IDENTIFY OVERLAP OF 16 BRIDGE SAMPLES 
overlap_samples <- intersect((MLB1_df1 %>% filter(!grepl("control", SampleID, ignore.case=T)))$SampleID,
                             (ProD_df1 %>% filter(!grepl("control", SampleID, ignore.case=T)))$SampleID)

# Find overlap in bridge samples, but exclude Olink controls

ProD_df1 <- ProD_df1 %>%
  select(-c('BXID'))

## BRIDGING NORMALIZATION
bridged_df <- OlinkAnalyze::olink_normalization(df1 = MLB1_df1,
                                                df2 = ProD_df1,
                                                overlapping_samples_df1 = overlap_samples,
                                                df1_project_nr = '20202249',
                                                df2_project_nr = 'BX0418',
                                              reference_project = '20202249')


## QC PLOT  
OlinkAnalyze::olink_qc_plot(bridged_df)

## RENAME PROTEINS & PANELS
qc_df <- bridged_df %>% 
  dplyr::mutate(Assay = gsub("-", "", Assay), # Remove all "-"
                Assay = gsub(" ", "_", Assay), # Changes spaces to "_"
                Assay = gsub("alpha", "a", Assay), # Rename alpha
                Assay = gsub("beta", "b", Assay), # Rename beta
                Assay = gsub("gamma", "g", Assay), # Rename gamma 
                Assay = gsub("IgG_Fc_receptor_IIb", "IgG_FcRIIb", Assay), # Shorten protein name
                Assay = gsub("PDGF_subunit_B", "PDGF_B", Assay)) %>% 
  dplyr::mutate(Panel = case_when(Panel == "Olink Cardiovascular II" ~ "CVDII", # Shorten panel name
                                  Panel == "Olink Inflammation" ~ "INF",
                                  TRUE ~ Panel))

## CALCULATE GLOBAL MISSING FREQUENCY
qc_df2 <- qc_df %>%
  group_by(Assay, Panel) %>%
  dplyr::mutate(MissingFreq = (as.numeric(MissingFreq))) %>%
  dplyr::mutate(no_missing_obs = sum(is.na(NPX) | NPX < LOD),
                total_obs = length(NPX)) %>%
  dplyr::mutate(GlobMissFreq = no_missing_obs / total_obs) %>% # This calculates the global MissingFreq across both batches
  ungroup()


## CHECK DUPLICATES
qc_dup <- qc_df2 %>%
  group_by(SampleID) %>%
  tally() # There will be duplicates for bridge and control samples


## SAVE QC DATA IN LONG FORMAT
write.table(qc_df2,
            file = "H:/From_SUND/Projects/ProDoc/ProDoc_Olink_bridged_QC_long.txt",
            sep = "\t",
            quote = F)




## CONVERT TO WIDE FORMAT (in case this format is preferred)
wide_df <- qc_df2 %>%
  dplyr::mutate(batch = str_extract(PlateID, '[^-]\\w+')) %>%
  dplyr::filter(QC_Warning != "Warning") %>% # Removes QC warning
  dplyr::filter(!str_detect(SampleID, 'Bridge')) %>% # Removes bridge samples
  dplyr::select(SampleID, Assay, NPX, QC_Warning) %>%
  pivot_wider(names_from = Assay,
              values_from = NPX) ## 33 ids failed INF, 25 failed CVDII

## SAVE QC DATA IN WIDE FORMAT
write.table(wide_df,
            file = "H:/From_SUND/Projects/ProDoc/ProDoc_Olink_bridged_QC_wide.txt",
            sep = "\t",
            quote = F)




## PANEL METATDATA
metadata <-
  qc_df2 %>% # using the df in the long format
  dplyr::group_by(Assay) %>%
  dplyr::select(UniProt,
                OlinkID,
                LOD,
                Panel,
                Panel_Version,
                MissingFreq,
                GlobMissFreq) %>% # These I think would be important to include
  dplyr::distinct(Assay, .keep_all = TRUE) %>% # This keeps one value for each protein in the long format
  arrange(Assay) # Order proteins


## SAVE METADATA
write.table(metadata,
            file = "H:/From_SUND/Projects/ProDoc/metadata.txt",
            sep = "\t",
            quote = F)





