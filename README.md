# sec-certs

Analyzer of security certificates (Common Criteria, NIST FIPS140-2...) 

## Usage

Steps:
 1. Run `search_certificate.py` to generate download scripts (download_cc_web.bat, download_cc_web.sh)
 2. Run `download_cc_web.bat` to download important files from Common Criteria website (requires `curl` installed)
 3. Run `search_certificate.py` to generate download scripts for separate pdf files with certificates (`download_active_certs.bat`...)
 4. Place all dowload scripts into folder on disk with at least 5GB free space (yes, there are A LOT of certificates) and run them 
 5. Run `search_certificate.py` to process CSV, HTML and PDF files. Extracted information is stored in *.json files. The consolidated info is in `certificate_data_complete.json`
 6. Run `search_certificate.py` to analyze information extracted to `certificate_data_complete.json`. Information like graph of certificates dependency is produced

## Extending the analysis

The analysis can be extended in several ways:
 1. Additional keywords can be extracted from PDF files (modify `cert_rules.py`)
 2. Data from `certificate_data_complete.json` can be analyzed in novel way - this is why this project was concieved at first place