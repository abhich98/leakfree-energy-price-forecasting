DATA_MANIFEST_VERSION = "1.0"
ML_REPORT_VERSION = "1.0"
# The date Germany switched from hourly to quarter-hourly electricity pricing is 1st October 2025 i.e, 2025-10-01. But, for regularity in our data processing, we consider the start date as 6th October 2025, which is the first Monday after the switch.
QUARTER_HOURLY_START_DATE = "2025-10-06"

# DATA PREPROCESSING PARAMETERS
FFILL_LIMIT = 3 # Maximum number of consecutive missing values to forward-fill. It also aligns with the hourly to quarter-hourly transition.