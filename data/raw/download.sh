#!/bin/bash

# Compensation
# Download file before executing from http://appsso.eurostat.ec.europa.eu/nui/setupDownloads.do
unzip nama_10_lp_ulc.zip

# Education
# https://ec.europa.eu/eurostat/web/products-datasets/-/trng_lfs_02
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/trng_lfs_02.tsv.gz
gunzip trng_lfs_02.tsv.gz

# Population (number)
# https://ec.europa.eu/eurostat/web/products-datasets/-/tps00001
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tps00001.tsv.gz
gunzip tps00001.tsv.gz

# Gross domestic expenditure
# https://ec.europa.eu/eurostat/web/products-datasets/-/t2020_20
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/t2020_20.tsv.gz
gunzip t2020_20.tsv.gz

# Total R&D personnel by sectors of performance, occupation and sex
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/rd_p_persocc.tsv.gz
gunzip rd_p_persocc.tsv.gz

# Fertility rates by age
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/demo_frate.tsv.gz

# R&D expenditure by source of funds (percentage)
# https://ec.europa.eu/eurostat/web/products-datasets/-/tsc00031
#curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tsc00031.tsv.gz
#gunzip tsc00031.tsv.gz


# Population by age group (percentage)
# https://ec.europa.eu/eurostat/web/products-datasets/-/tps00010
#curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tps00010.tsv.gz
#gunzip tps00010.tsv.gz
