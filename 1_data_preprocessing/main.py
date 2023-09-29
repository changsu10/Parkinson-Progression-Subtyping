import extraction
import patient
import patient_longitudinal_info
import format

version = "xxxxxx" # please specify version (folder name) of your data
output = "processed_data"



patient.extract_patient_info(version, output)

patient_longitudinal_info.extract_longitudinal_info(output)

extraction.extract_data(version, output)

mputation_method = "interpolate"  # options: interpolate, LOCF&FOCB

format.format(output, 'interpolate')

format.format(output, 'LOCF&FOCB')
