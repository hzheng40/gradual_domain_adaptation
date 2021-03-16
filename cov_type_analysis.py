import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_covtype_data(load_file, normalize=True, metric='water'):
    df = pd.read_csv(load_file, header=None)
    data = df.to_numpy()
    xs = data[:, :54]
    if normalize:
        xs = (xs - np.mean(xs, axis=0)) / np.std(xs, axis=0)
    ys = data[:, 54] - 1

    # Keep the first 2 types of crops, these comprise majority of the dataset.
    keep = (ys <= 1)
    print(len(xs))
    xs = xs[keep]
    ys = ys[keep]
    print(len(xs))

    if metric == 'roadway':
        # Sort by (horizontal) distance to roadway.
        dist_to_roadway = xs[:, 5]
        indices = np.argsort(dist_to_roadway, axis=0)
        xs = xs[indices]
        ys = ys[indices]
    elif metric == 'firepoint':
        # Sort by (horizontal) distance to firepoint.
        dist_to_firepoint = xs[:, 3]
        indices = np.argsort(dist_to_firepoint, axis=0)
        xs = xs[indices]
        ys = ys[indices]
    else:
        # Sort by (horizontal) distance to water body.
        dist_to_water = xs[:, 3]
        indices = np.argsort(dist_to_water, axis=0)
        xs = xs[indices]
        ys = ys[indices]
    return xs, ys

if __name__ == '__main__':
    df = pd.read_csv('data/covtype.data', header=None)
    data = df.to_numpy()
    data_roadway = data[:, 5]
    data_firepoint = data[:, 9]
    data_water = data[:, 3]

    plt.rcParams.update({'font.size': 32})

    # binning and plotting as bars

    # water
    water_min = np.min(data_water)
    water_max = np.max(data_water)
    water_bins = np.linspace(water_min, water_max, num=100)
    water_digitized = np.digitize(data_water, bins=water_bins)
    water_binned = np.bincount(water_digitized)
    if water_binned.shape[0] > water_bins.shape[0]:
        water_binned = water_binned[:100]
    bin_width = int(water_bins[1] - water_bins[0]) - 2
    plt.bar(water_bins, water_binned, bin_width, align='edge')
    plt.xlabel('Horizontal Distance to Hydrology')
    plt.ylabel('Count')
    plt.title('Cover Type: Hydrology')
    plt.ylim(0, 50000)
    plt.show()

    # fire
    fire_min = np.min(data_firepoint)
    fire_max = np.max(data_firepoint)
    fire_bins = np.linspace(fire_min, fire_max, num=100)
    fire_digitized = np.digitize(data_firepoint, bins=fire_bins)
    fire_binned = np.bincount(fire_digitized)
    if fire_binned.shape[0] > fire_bins.shape[0]:
        fire_binned = fire_binned[:100]
    bin_width = int(fire_bins[1] - fire_bins[0]) - 8
    plt.bar(fire_bins, fire_binned, bin_width, align='edge')
    plt.xlabel('Horizontal Distance to Fire Points')
    plt.ylabel('Count')
    plt.title('Cover Type: Fire Points')
    plt.ylim(0, 50000)
    plt.show()

    # roadway
    road_min = np.min(data_roadway)
    road_max = np.max(data_roadway)
    road_bins = np.linspace(road_min, road_max, num=100)
    road_digitized = np.digitize(data_roadway, bins=road_bins)
    road_binned = np.bincount(road_digitized)
    if road_binned.shape[0] > road_bins.shape[0]:
        road_binned = road_binned[:100]
    bin_width = int(road_bins[1] - road_bins[0]) - 8
    plt.bar(road_bins, road_binned, bin_width, align='edge')
    plt.xlabel('Horizontal Distance to Roadways')
    plt.ylabel('Count')
    plt.title('Cover Type: Roadways')
    plt.ylim(0, 50000)
    plt.show()