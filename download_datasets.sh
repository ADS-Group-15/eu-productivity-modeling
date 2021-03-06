#!/bin/bash

cd data/raw

# Compensation
# Download file before executing from https://ec.europa.eu/eurostat/tgm/graph.do?tab=graph&plugin=1&pcode=tesem160&language=en&toolbox=sort
unzip nama_10_lp_ulc.zip

# Education
# https://ec.europa.eu/eurostat/web/products-datasets/-/trng_lfs_02
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/trng_lfs_02.tsv.gz

# Population (number)
# https://ec.europa.eu/eurostat/web/products-datasets/-/tps00001
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tps00001.tsv.gz

# Gross domestic expenditure
# https://ec.europa.eu/eurostat/web/products-datasets/-/t2020_20
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/t2020_20.tsv.gz

# Total R&D personnel by sectors of performance, occupation and sex
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/rd_p_persocc.tsv.gz

# Fertility rates by age
# https://ec.europa.eu/eurostat/web/products-datasets/-/demo_frate
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/demo_frate.tsv.gz

# Human resources in science and technology (HRST)
# https://ec.europa.eu/eurostat/web/products-datasets/-/tsc00025
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tsc00025.tsv.gz

# Tax rate
# https://ec.europa.eu/eurostat/web/products-datasets/-/earn_nt_taxrate
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/earn_nt_taxrate.tsv.gz

# Imports of goods and services in % of GDP
# https://ec.europa.eu/eurostat/web/products-datasets/-/tet00004
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tet00004.tsv.gz

# International trade, by reporting country, total product
# https://ec.europa.eu/eurostat/web/products-datasets/-/tet00002
curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tet00002.tsv.gz

# Gross domestic product
# https://ec.europa.eu/eurostat/web/products-datasets/-/teina011
#curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/teina011.tsv.gz

# Funding of education
# https://ec.europa.eu/eurostat/web/products-datasets/-/educ_fifunds
#curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/educ_fifunds.tsv.gz

# R&D expenditure by source of funds (percentage)
# https://ec.europa.eu/eurostat/web/products-datasets/-/tsc00031
#curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tsc00031.tsv.gz

# Population by age group (percentage)
# https://ec.europa.eu/eurostat/web/products-datasets/-/tps00010
#curl -OL https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tps00010.tsv.gz
