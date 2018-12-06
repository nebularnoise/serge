import argparse
from src import arch

parser = argparse.ArgumentParser(description="Utility for training a custom \
VAE based on input arguments")

parser.add_argument("--determinist_encoder", type=bool, default=False,
                    help="Defines wheither the encoder is determinist or not")


args = parser.parse_args()
