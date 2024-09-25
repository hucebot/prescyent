import h5py

import prescyent.dataset.features as p_features


def load_features(h_file: h5py.File):
    features = []
    for feat_name in h_file["tensor_features"].keys():
        feat = h_file["tensor_features"][feat_name]
        features.append(
            getattr(p_features, feat.attrs["feature_class"])(
                ids=list(feat),
                name=feat_name,
                distance_unit=feat.attrs["distance_unit"],
            )
        )
    return p_features.Features(features, index_name=False)


def write_metadata(h_file: h5py.File, metadata):
    """write the metadata of a dataset into the hdf5 file"""
    h_file.attrs["frequency"] = metadata.BASE_FREQUENCY
    h_file.attrs["point_parents"] = metadata.POINT_PARENTS
    h_file.attrs["point_names"] = metadata.POINT_LABELS
    tensor_features = h_file.create_group("tensor_features")
    for feat in metadata.DEFAULT_FEATURES:
        hdf_feat = tensor_features.create_dataset(feat.name, data=feat.ids)
        hdf_feat.attrs["distance_unit"] = feat.distance_unit
        hdf_feat.attrs["feature_class"] = feat.__class__.__name__


def get_dataset_keys(h_group: h5py.Group):
    """return list of keys for each dataset inside the h5py group and subgroups"""
    keys = []
    h_group.visit(
        lambda key: keys.append(key) if isinstance(h_group[key], h5py.Dataset) else None
    )
    return keys
