# Copy to lasair_configs.py and fill in your Lasair credentials.
# lasair_configs.py is gitignored — do not commit secrets.

API_TOKEN = 'your-lasair-api-token'
LASAIR_ENDPOINT = 'https://lasair-ztf.lsst.ac.uk/api'
TOPIC_IN = 'lasair_750NEEDLEINPUTALERTS'
GROUP_ID = 'test_YYYYMMDD'
TOPIC_OUT = 'NEEDLE'
TEST = True  # skip george/GP_fitting imports (Lasair streaming does not need them)
# MAPSDIR defaults to <project>/source/maps (must contain maps.yaml and lambda_sfd_ebv.fits)
