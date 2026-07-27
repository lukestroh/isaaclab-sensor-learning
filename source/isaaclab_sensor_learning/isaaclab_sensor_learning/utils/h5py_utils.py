#!/usr/bin/env python3
import h5py as h5


def h5_to_dict(node) -> dict:
    """
    Convert an h5py File/Group/Dataset object to a dictionary, preserving attrs.
    """
    if hasattr(node, "keys"):
        # Group node — recurse into children, then attach this group's own attrs
        result = {key: h5_to_dict(node[key]) for key in node.keys()}
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