import argparse

from bno.datasets import MNISTHelmholtz


def test_mnisthelmholtz_dataset():
  dataset = MNISTHelmholtz(
    image_size=128,
    pml_size=32,
    sound_speed_lims=[1., 1.5],
    source_pos=(96, 48),  # In pixels
    omega=1.0,
    num_samples=10,
    regenerate=True
  )

def plot_mnisthelmholtz_dataset():
  from matplotlib import pyplot as plt

  dataset = MNISTHelmholtz(
    image_size=128,
    pml_size=32,
    sound_speed_lims=[1., 1.5],
    source_pos=(96, 48),  # In pixels
    omega=1.0,
    num_samples=10,
    regenerate=False
  )

  fig, ax = plt.subplots(1, 2)
  ax[0].imshow(dataset[5]['field'].real, vmin=-0.05, vmax=0.05, cmap="RdBu_r")
  ax[1].imshow(dataset[5]['sound_speed'], cmap="inferno")
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", action="store_true")
  parser.add_argument("--plot", action="store_true")
  args = parser.parse_args()

  if args.test:
    test_mnisthelmholtz_dataset()
  if args.plot:
    plot_mnisthelmholtz_dataset()
