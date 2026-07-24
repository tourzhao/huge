# Compare every exported R function's signature against its Rd \arguments.
# Reports functions whose formals are undocumented or whose docs mention
# arguments that no longer exist. Run from the repo root:
#   Rscript tools/check_doc_signatures.R
suppressMessages(library(tools))
suppressMessages(library(huge))

db <- Rd_db(dir = ".")
exports <- getNamespaceExports("huge")
ns <- asNamespace("huge")
funs <- sort(exports[vapply(exports, function(f) is.function(get(f, envir = ns)), logical(1))])

rd_aliases <- lapply(db, function(rd)
  unlist(lapply(rd, function(x)
    if (identical(attr(x, "Rd_tag"), "\\alias")) as.character(x))))

extract_doc_args <- function(rd) {
  args_doc <- character(0)
  walk <- function(x) {
    if (is.list(x)) {
      if (identical(attr(x, "Rd_tag"), "\\arguments")) {
        for (it in x)
          if (identical(attr(it, "Rd_tag"), "\\item"))
            args_doc <<- c(args_doc, paste(unlist(it[[1]]), collapse = ""))
      } else lapply(x, walk)
    }
  }
  walk(rd)
  gsub("\\\\dots", "...", args_doc)
}

status <- 0
for (f in funs) {
  fml <- names(formals(get(f, envir = ns)))
  hit <- NULL
  for (nm in names(db)) if (f %in% rd_aliases[[nm]]) { hit <- nm; break }
  if (is.null(hit)) {
    cat(sprintf("NO-RD    %s\n", f)); status <- 1; next
  }
  args_doc <- extract_doc_args(db[[hit]])
  undocumented <- setdiff(setdiff(fml, "..."), args_doc)
  stale <- setdiff(setdiff(args_doc, "..."), fml)
  if (length(undocumented) || length(stale)) {
    cat(sprintf("MISMATCH %-16s undocumented: [%s]  stale-doc: [%s]\n",
                f, paste(undocumented, collapse = ","),
                paste(stale, collapse = ",")))
    status <- 1
  }
}
if (status == 0) cat("all exported signatures match their documentation\n")
quit(status = status)
