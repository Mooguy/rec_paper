library(GOSemSim)
library(igraph)
library(GO.db)
library(org.Mm.eg.db)
library(jsonlite)

# --- NEW: Capture arguments from Python ---
args <- commandArgs(trailingOnly = TRUE)
# args[1] will be the input path, args[2] will be the output path
input_path <- args[1]
output_path <- args[2]

# Prepare semantic data
mmGO_list <- list(
  BP = godata('org.Mm.eg.db', ont = "BP"),
  MF = godata('org.Mm.eg.db', ont = "MF"),
  CC = godata('org.Mm.eg.db', ont = "CC")
)

# Load GO lists
go_lists <- fromJSON(input_path)

# Function to process a list of GO IDs
process_go_list <- function(go_ids, mmGO_list) {
  if (length(go_ids) == 0) {
    return(list(
      valid_ids = character(),
      labels = list(),
      ontologies = list(),
      sim_matrix = matrix(numeric(0), nrow = 0, ncol = 0),
      invalid_ids = character()
    ))
  }
  
  # Keep only valid IDs
  valid_ids <- go_ids[go_ids %in% keys(GOTERM)]
  
  # Get labels and ontology assignment
  labels <- Term(GOTERM[valid_ids])
  ont_map <- Ontology(GOTERM[valid_ids])  # "BP","MF","CC"
  
  # Initialize a big similarity matrix with NA
  sim_matrix <- matrix(NA, nrow = length(valid_ids), ncol = length(valid_ids))
  rownames(sim_matrix) <- valid_ids
  colnames(sim_matrix) <- valid_ids
  
  # Compute similarity within each ontology
  for (ont in unique(ont_map)) {
    ids <- valid_ids[ont_map == ont]
    if (length(ids) > 1) {
      semData <- mmGO_list[[ont]]
      m <- mgoSim(ids, ids, semData = semData, measure = "Wang", combine = NULL)
      m <- as.matrix(m)
      sim_matrix[ids, ids] <- m[ids, ids]
    } else if (length(ids) == 1) {
      sim_matrix[ids, ids] <- 1  # self-similarity
    }
  }
  
  return(list(
    valid_ids = valid_ids,
    labels = as.list(labels),
    ontologies = as.list(ont_map),
    sim_matrix = sim_matrix,
    invalid_ids = setdiff(go_ids, valid_ids)
  ))
}

# Run processing
results <- lapply(go_lists, process_go_list, mmGO_list = mmGO_list)

# Convert results to JSON-safe format
results_json <- lapply(results, function(x) {
  list(
    valid_ids = x$valid_ids,
    labels = x$labels,
    ontologies = x$ontologies,
    sim_matrix = list(
      data = unname(x$sim_matrix),
      rownames = rownames(x$sim_matrix),
      colnames = colnames(x$sim_matrix)
    ),
    invalid_ids = x$invalid_ids
  )
})

new_dir_path = 'data/processed/go_analysis/go_jsons/'

# Write to JSON
if (!dir.exists(new_dir_path)) {
  dir.create(new_dir_path, recursive = TRUE) 
} else {
  print("Directory already exists.")
}
write_json(results_json, output_path, pretty = TRUE)


