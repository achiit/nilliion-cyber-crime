import os
from dotenv import load_dotenv

load_dotenv()

# replace this with your program_id
CONFIG_PROGRAM_NAME = "healthcare_imaging_compute"

CONFIG_NUM_PARAMS = 31

# 1st party
CONFIG_TEST_PARTY_1 = {
    "userkey_file": os.getenv("NILLION_USERKEY_PATH_PARTY_1"),
    "nodekey_file": os.getenv("NILLION_NODEKEY_PATH_PARTY_1"),
    "nodekey_alternate_file": os.getenv("NILLION_NODEKEY_PATH_PARTY_4"),
    "party_name": "Party1",
}

# N other parties
CONFIG_HP_PARTIES = [
    {
        "userkey_file": os.getenv("NILLION_USERKEY_PATH_PARTY_2"),
        "nodekey_file": os.getenv("NILLION_NODEKEY_PATH_PARTY_2"),
        "party_name": "Party2",
        "dataset": "scaled_theta_subset_sm",
        "secret_name": "hp1_p",
    },
    {
        "userkey_file": os.getenv("NILLION_USERKEY_PATH_PARTY_3"),
        "nodekey_file": os.getenv("NILLION_NODEKEY_PATH_PARTY_3"),
        "party_name": "Party3",
        "dataset": "scaled_theta_subset_lg",
        "secret_name": "hp2_p",
    },
]
