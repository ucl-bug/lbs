from bno import datasets


def main():

  for max_sos in [1.2, 1.5, 2.0]:
    print(f"Generating MNIST dataset for sound speed limit of {max_sos}\n")
    _ = datasets.MNISTHelmholtz(
        image_size=128,
        pml_size=32,
        sound_speed_lims=[1., max_sos],
        source_pos=(96, 48),  # In pixels
        omega=1.0,
        num_samples=5000,
        regenerate=True
      )

if __name__ == '__main__':
  main()
