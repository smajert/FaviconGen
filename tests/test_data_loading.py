import numpy as np
from matplotlib import pyplot as plt

import favicon_gen.data_loading as dlo


def test_all_files_found(LogoDatasetLocation):
    file_loader = dlo.LargeLogoDataset(LogoDatasetLocation)
    assert len(file_loader) == 486377


def test_labels_correct_when_restricting_to_cluster(LogoDatasetLocation):
    n_images, file_loader = dlo.load_logos(200, False, None, [2, 20, 50])
    labels_concat = []
    for batch, labels in file_loader:
        labels_concat.append(labels)

    collected_labels = np.concatenate(labels_concat)
    assert n_images == 11650
    assert np.max(collected_labels) == 2
    assert np.min(collected_labels) == 0


def test_image_grid_and_loading(LogoDatasetLocation):
    lld = dlo.load_logos(64, shuffle=True, n_images=None, clusters=[2])[1]
    batch = next(iter(lld))[0]

    # mnist = testee.load_mnist(64, shuffle=True, n_images=None)[1]
    # batch = next(iter(mnist))[0]

    dlo.show_image_grid(batch)
    show_plot = False
    if show_plot:
        plt.show()
