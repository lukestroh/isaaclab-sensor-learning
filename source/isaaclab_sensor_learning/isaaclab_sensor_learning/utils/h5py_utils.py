#!/usr/bin/env python3
import h5py as h5


def h5_to_dict(node, attrs_only: tuple[str, ...] = ()) -> dict:
    """
    Convert an h5py File/Group/Dataset object to a dictionary, preserving attrs.

    :param attrs_only: Child group names whose datasets are skipped; only their (and their subgroups') attrs are read.
    """
    if hasattr(node, "keys"):
        # Group node — recurse into children, then attach this group's own attrs
        result = {
            key: h5_attrs_to_dict(node[key]) if key in attrs_only else h5_to_dict(node[key], attrs_only)
            for key in node.keys()
        }
        attrs = {attr: node.attrs[attr] for attr in node.attrs}
        if attrs:
            result["attrs"] = attrs
        return result
    elif hasattr(node, "read_direct"):
        # Dataset node — read into numpy array
        value = node[()]  # use [()] instead of [:] — [:] fails on scalar datasets
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif hasattr(value, "dtype") and value.dtype.kind == "S":
            value = value.astype(str)
        attrs = {attr: node.attrs[attr] for attr in node.attrs}
        return {"data": value, "attrs": attrs} if attrs else value
    else:
        return node


def h5_attrs_to_dict(node) -> dict:
    """
    Read only the attrs of an h5py Group and its subgroups. Datasets are not loaded.
    """
    result = {key: h5_attrs_to_dict(node[key]) for key in node.keys() if hasattr(node[key], "keys")}
    attrs = {attr: node.attrs[attr] for attr in node.attrs}
    if attrs:
        result["attrs"] = attrs
    return result
