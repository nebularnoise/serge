import argparse
from src import arch

parser = argparse.ArgumentParser(description="Utility for training a custom \
VAE based on input arguments")

parser.add_argument("--deter", action="store_true",
                    help="Defines wheither the encoder is determinist or not")
parser.add_argument("--zdim", type=int, default=32,
                    help="Dimension of the latent space")
args = parser.parse_args()

model = arch.VariationnalAutoEncoder(determinist_encoder=args.deter,
                                    z_dim=args.zdim)
