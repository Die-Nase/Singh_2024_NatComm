import tifffile
from pathlib import Path
import codecs
import numpy as np
import copy
import pandas as pd
import pint
import pint_pandas
from skimage import measure
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator


class FileReader:
    def __init__(self):
        pass
    
    @classmethod
    def select_file_reader(cls, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError("no such file in directory")

        if filepath.suffix == '.tif' or filepath.suffix == '.tiff':
            return TiffReader()
        
        elif filepath.suffix == '.npy':
            return LabelsReader()
            
        else:
            raise NotImplementedError(
                "FileReader for format {} is not implemented".format(filepath.suffix))
            
    @classmethod
    def to_LayerDataTuple(cls, filepath, time_unit = None, space_unit = None):
        self = cls()
        reader = self.select_file_reader(filepath)
        layer_data_tuple = reader.to_LayerDataTuple(filepath, time_unit = time_unit, space_unit = space_unit)
        return layer_data_tuple


class LabelsReader:
    def read(self, filepath, space_unit=None, time_unit=None, required_dims = None):
        filepath = Path(filepath)
        arr = self.read_data(filepath)
        kwargs = self.read_metadata(filepath, space_unit=space_unit, time_unit=time_unit)
        
        if "shape" in kwargs:
            shape = kwargs.pop("shape")
            arr = arr.reshape(shape)
        if required_dims:
            #arr, new_kwargs = reorder_dims(arr, kwargs["dims"], required_dims, kwargs["scale"], kwargs["units"], kwargs["origin"])
            arr, new_kwargs = reorder_dimensions(arr, kwargs["dims"], required_dims, kwargs)
            kwargs.update(new_kwargs)
        
        kwargs["filepath"] = filepath
        return arr, kwargs
    
    def read_data(self, filepath):
        arr = self.read_npy(filepath)
        return arr
    
    def read_metadata(self, filepath, space_unit = None, time_unit = None):
        kwargs = self.read_npy_meta(filepath, space_unit=space_unit, time_unit=time_unit)
        return kwargs
    
    def read_npy(self, filepath):
        seg_dict = np.load(filepath, allow_pickle = True).item()
        arr = seg_dict["masks"]
        return arr
    
    def read_npy_meta(self, filepath, space_unit = None, time_unit = None):
        filedir = filepath.parent
        corresponding_tiffile = filedir.joinpath(filepath.stem.split("_seg")[0]+".tif")
        
        if corresponding_tiffile.exists():
            tif, kwargs = TiffReader().read(corresponding_tiffile, space_unit=space_unit, time_unit=time_unit, lazy_loading = False)
        else:
            raise FileNotFoundError("{} does not exist".format(str(corresponding_tiffile)))
        
        kwargs["filepath"] = filepath
        kwargs["name"] = filepath.stem
        
        shape = np.array(tif.shape)
        for i,dim in enumerate(kwargs["dims"]):
            if dim == "c":
                shape[i] = 1
                kwargs["shape"] = tuple(shape)
                break
            
        return kwargs
        
    def to_LayerDataTuple(self, filepath, space_unit=None, time_unit=None, lazy_loading=False):
        
        data, kwargs = self.read(filepath, space_unit=space_unit, time_unit=time_unit, required_dims="tzyx")
        kwargs = to_napari_layer_kwargs(kwargs)
        return [(data, kwargs, "Labels")]
        
        
class TiffReader:
    
    def read(self, filepath, space_unit=None, time_unit=None, required_dims = None, lazy_loading=False):
        filepath = Path(filepath)
        with tifffile.TiffFile(filepath) as tif:
            # get metadata
            tif_tags = self.get_tif_tags(tif)
            name = tif.filename.split(".")[0]
            
            if tif.is_imagej:
                shape, scale, units, dims, origin = get_scale_from_imagej_metadata(
                    tif_tags, space_unit=space_unit, time_unit=time_unit, reduced_dims=False)
                kwargs = {"name": name, "scale": scale, "origin": origin, 'filepath': filepath, "units": units, "dims": dims}
            else:
                raise NotImplementedError("support for this metadata type is not implemented")

            # get data
            data = tif.asarray()
        
        data = data.reshape(shape)
        
        if required_dims:
            #data, kwargs = reorder_dims(data, dims, required_dims, scale = scale, units = units, origin = origin)
            data, kwargs = reorder_dimensions(data, kwargs["dims"], required_dims, kwargs)
            kwargs["filepath"] = filepath
            kwargs["name"] = name
    
        return data, kwargs

    def read_data(self, filepath):
        filepath = Path(filepath)
        with tifffile.TiffFile(filepath) as tif:
            return tif.asarray()

    def read_tif_tags(self, filepath):
        filepath = Path(filepath)
        with tifffile.TiffFile(filepath) as tif:
            tif_tags = self.get_tif_tags(tif)
            return tif_tags

    def get_tif_tags(self, tif):
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
        return tif_tags

    def to_LayerDataTuple(self, filepath, time_unit, space_unit):
        data, kwargs = self.read(filepath, space_unit=space_unit,
                                 time_unit=time_unit, required_dims = "tczyx")
        
        kwargs = to_napari_layer_kwargs(kwargs)
        
        t, c, z, y, x = data.shape
        
        layer_data_tuples = []
        for i in range(c):
            kwargs_channel = kwargs_drop_channel(kwargs, i)
            layer_data_tuple = (data[:,i,:,:,:], kwargs_channel, "Image")
            layer_data_tuples.append(layer_data_tuple)
        return layer_data_tuples


def get_scale_from_imagej_metadata(metadata, space_unit=None,
                                   time_unit=None, reduced_dims=True):

    image_description = metadata["ImageDescription"]
    if "ImageJ" in image_description:
        image_desc_items = image_description.split("\n")
        image_desc_dict = {}
        for item in image_desc_items:
            key, value = item.split("=")
            image_desc_dict[key] = value

        # get dimension units
        x_scale_unit = image_desc_dict['unit']
        y_scale_unit = image_desc_dict.get('yunit', x_scale_unit)
        z_scale_unit = image_desc_dict.get('zunit', x_scale_unit)
        t_scale_unit = image_desc_dict.get('tunit', "sec")
        if t_scale_unit == "h":
            t_scale_unit = "hour"

        # decode if units are given in unicode
        if "\\" in x_scale_unit:
            x_scale_unit = codecs.decode(x_scale_unit, 'unicode_escape')

        if "\\" in y_scale_unit:
            y_scale_unit = codecs.decode(y_scale_unit, 'unicode_escape')

        if "\\" in z_scale_unit:
            z_scale_unit = codecs.decode(z_scale_unit, 'unicode_escape')

        if "\\" in t_scale_unit:
            t_scale_unit = codecs.decode(t_scale_unit, 'unicode_escape')

        # get scale magnitudes
        x_res = metadata["XResolution"]
        x_scale_magnitude = x_res[1]/x_res[0]
        y_res = metadata["YResolution"]
        y_scale_magnitude = y_res[1]/y_res[0]
        z_scale_magnitude = float(
            image_desc_dict.get("spacing", x_scale_magnitude))
        t_scale_magnitude = float(image_desc_dict.get("finterval", 0))
        
        # get origin pixel
        x_origin_pixel = float(image_desc_dict.get('xorigin', 0))
        y_origin_pixel = float(image_desc_dict.get('yorigin', 0))
        z_origin_pixel = float(image_desc_dict.get('zorigin', 0))

        x_scale = pint.Quantity(x_scale_magnitude, x_scale_unit)
        y_scale = pint.Quantity(y_scale_magnitude, y_scale_unit)
        z_scale = pint.Quantity(z_scale_magnitude, z_scale_unit)
        t_scale = pint.Quantity(t_scale_magnitude, t_scale_unit)
        

        # transform to according to scale bar unit
        if space_unit:
            x_scale = x_scale.to(space_unit)
            y_scale = y_scale.to(space_unit)
            z_scale = z_scale.to(space_unit)

        if time_unit:
            t_scale = t_scale.to(time_unit)

        # check dimensionality of hyperstack
        t = int(image_desc_dict.get('frames', "1"))
        c = int(image_desc_dict.get('channels', "1"))
        z = int(image_desc_dict.get('slices', "1"))
        y = metadata["ImageLength"]
        x = metadata["ImageWidth"]

        # dimensions, scale and units of 5D Image
        shape = (t, z, c, y, x)
        dims = "".join(["t", "z", "c", "y", "x"])
        scale = (t_scale.magnitude, z_scale.magnitude, 1, y_scale.magnitude,
                 x_scale.magnitude)
        units = (str(t_scale.units), str(z_scale.units),
                 str(pint.Quantity(1, "").units), str(y_scale.units), str(x_scale.units))
        
        origin = (0,
                  z_origin_pixel * z_scale.magnitude,
                  0,
                  y_origin_pixel * y_scale.magnitude,
                  x_origin_pixel * x_scale.magnitude,)
        
        n_images = int(image_desc_dict['images'])
        if (t*c*z) != n_images:
            raise Exception("""dimensionality of stack: contradicting metadata\n
                            t*c*z is not equal to number of images""")

        # return reduced 5D dims, scale and units to used dimensionalties
        shape_reduced = []
        dims_reduced = []
        scale_reduced = []
        units_reduced = []
        origin_reduced = []

        for i, axis_length in enumerate(shape):
            if axis_length > 1:
                shape_reduced.append(shape[i])
                scale_reduced.append(scale[i])
                units_reduced.append(units[i])
                origin_reduced.append(origin[i])
                dims_reduced.append([*dims][i])

        # return results
        if reduced_dims:
            return tuple(shape_reduced), tuple(scale_reduced), tuple(units_reduced), "".join(dims_reduced), tuple(origin_reduced)
        return shape, scale, units, dims, origin

    else:
        raise Exception("scale could not be found in metadata")


def reorder_dimensions(input_array, original_dims: str, new_dims: str, metadata={}):
    """
    Reorders dimensions of a numpy array and updates associated metadata.

    Parameters:
    - input_array (numpy.ndarray): The input array to be reordered.
    - original_dims (str): The original order of dimensions as a string.
    - new_dims (str): The desired order of dimensions as a string.
    - metadata (dict, optional): Additional metadata associated with dimensions.
        - "scale" (list, optional): List of scaling factors for each dimension.
        - "units" (list, optional): List of units for each dimension.
        - "origin" (list, optional): List of origins for each dimension.

    Returns:
    - numpy.ndarray: The reordered array.
    - dict: Updated metadata based on the reordered dimensions.
    """
    metadata = copy.deepcopy(metadata)
    
    # Convert dimension strings to lists for manipulation
    original_dims_list = list(original_dims)
    new_dims_list = list(new_dims)
    
    # Check if all old dimensions with length > 1 are included in new_dims
    for dim, size in zip(original_dims_list, input_array.shape):
        if dim not in new_dims_list and size > 1:
            raise ValueError(f"All old 'dims' with a length > 1 have to be included in 'new_dims'. "
                             f"Length of {dim} is {size}, however it's not included in {new_dims}. "
                             f"To drop dimension with a length > 1 use indexing, for example arr = arr[:, 27, :]")
    
    # Create new axes if needed
    additional_dims = [new_dim for new_dim in new_dims_list if new_dim not in original_dims_list]
    current_shape = [1] * len(additional_dims) + list(input_array.shape)
    input_array = input_array.reshape(current_shape)
    current_dims_list = additional_dims + original_dims_list
    
    # Reorder axes using einsum
    current_dims_str = "".join(current_dims_list)
    new_dims_str = "".join(new_dims_list)
    output_array = np.einsum(f"{current_dims_str}->{new_dims_str}", input_array)
    
    # Update metadata based on the reordered dimensions
    new_metadata = {"scale": np.ones(len(new_dims_list)),
                   "units": ["dimensionless"]*len(new_dims_list),
                   "origin": np.zeros(len(new_dims_list))}
    if "metadata" in metadata:
        metadata.update(metadata.pop("metadata"))
    for i, dim in enumerate(original_dims_list):
        if dim in new_dims_list:
            idx = new_dims_list.index(dim)
            for key in ["scale", "units", "origin"]:
                if key in metadata:
                    new_metadata[key][idx] = metadata[key][i]
    
    for key in ["scale", "units", "origin"]:
        new_metadata[key] = tuple(new_metadata[key])
    
    # Create a new dictionary for updated metadata
    new_metadata["dims"] = new_dims_str
    for key in metadata:
        if key not in new_metadata:
            new_metadata[key] = metadata[key]

    return output_array, new_metadata

def to_napari_layer_kwargs(kwargs):
    kwargs = copy.deepcopy(kwargs)
    new_kwargs = {}

    if "origin" in kwargs:
        kwargs["translate"] = kwargs.pop("origin")
        
    primary_keys = ["name", "scale", "translate",]
    for key in primary_keys:
        if key in kwargs:
            new_kwargs[key] = kwargs.pop(key)
    
    new_kwargs["metadata"] = kwargs
    
    if "scale" in new_kwargs:
        scale = np.array(new_kwargs["scale"])
        for i,dim in enumerate(scale):
            if dim == 0:
                print("dim {} has a scale of 0. This is unvalid for a napari layer. Scale will be forced to 1".format(i))
                scale[i] = 1
        new_kwargs["scale"] = scale
    return new_kwargs
                
def kwargs_drop_channel(kwargs, channel_idx, channel_name = None):
    kwargs = copy.deepcopy(kwargs)

    "assumptions: dims = 'tczyx' & kwargs is a data_data_tuple kwargs"
    
    new_kwargs = {}
    new_kwargs["name"] = kwargs.get("name") + "_c"+"%04d" % channel_idx
    new_kwargs["metadata"] = {}
    new_kwargs["metadata"]["channel"] = channel_idx

    
    keys = ["scale", "translate"]
    for key in keys:
        if key in kwargs:
            attr = kwargs.pop(key)
            new_kwargs[key] = (attr[0], attr[2], attr[3], attr[4],)
    
    if "metadata" in kwargs:
        keys = ["units",]
        if key in kwargs["metadata"]:
            attr = kwargs["metadata"].pop(key)
            new_kwargs["metadata"][key] = (attr[0], attr[2], attr[3], attr[4])
        if "dims" in kwargs["metadata"]:
            dims = kwargs["metadata"].pop("dims")
            new_kwargs["metadata"]["dims"] = "".join([dims[0], dims[2], dims[3], dims[4]])
            
        
        new_kwargs["metadata"].update(kwargs["metadata"])
    
    return new_kwargs


properites_units = {
    "area": "pint[length_unit**sdim]",
    "area_bbox": "pint[length_unit**sdim]",
    "equivalent_diameter_area": "pint[length_unit]",
    "area_filled": "pint[length_unit**sdim]",
    "area_convex": "pint[length_unit**sdim]",
    "centroid_x": "pint[length_unit]",
    "centroid_y": "pint[length_unit]",
    "centroid_z": "pint[length_unit]",
    "centroid_local_x": "pint[length_unit]",
    "centroid_local_y":	"pint[length_unit]",
    "centroid_local_z":	"pint[length_unit]",
    "centroid_weighted_x": "pint[length_unit]",
    "centroid_weighted_y": "pint[length_unit]",
    "centroid_weighted_z": "pint[length_unit]",
    "centroid_weighted_local_x": "pint[length_unit]",
    "centroid_weighted_local_y": "pint[length_unit]",
    "centroid_weighted_local_z": "pint[length_unit]",
    "feret_diameter_max": "pint[length_unit]",
    "axis_major_length": "pint[length_unit]",
    "axis_minor_length": "pint[length_unit]",
    "orientation": "pint[radian]",
    "perimeter": "pint[length_unit]",
    "perimeter_crofton": "pint[length_unit]",
    "minor_axis_length": "pint[length_unit]",
    "major_axis_length": "pint[length_unit]"
}

properties3D = {
    "size_properties": ["area", "area_bbox", "equivalent_diameter_area", "area_filled"],
    "position_properties": ["centroid", "centroid_local", "centroid_weighted", "centroid_weighted_local"],
    "region_properties": ["bbox", "image", "image_convex", "image_filled", "image_intensity", "coords", "slice"],
    "shape_properties": ["euler_number", "extent", "feret_diameter_max", "axis_major_length", "axis_minor_length", "orientation", "solidity", "aspect_ratio"],
    "moment_properties": ["inertia_tensor", "inertia_tensor_eigvals", "moments", "moments_central", "moments_normalized", "moments_weighted", "moments_weighted_central", "moments_weighted_normalized"],
    "intensity_properties": ["intensity_max", "intensity_mean", "intensity_min"],
}

properties2D = copy.deepcopy(properties3D)
properties2D["size_properties"].extend(["area_convex"])
properties2D["shape_properties"].extend(["eccentricity", "perimeter", "perimeter_crofton", "roundness", "circularity","perimeter_area_ratio"])
properties2D["moment_properties"].extend(["moments_hu", "moments_weighted_hu"])


properties3D_table = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in properties3D.items() ]))
properties3D_table[properties3D_table.isnull()] = ""
properties3D_table = properties3D_table.style.set_table_attributes("style='display:inline'").set_caption('3D Shape Properties')

properties2D_table = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in properties2D.items() ]))
properties2D_table[properties2D_table.isnull()] = ""
properties2D_table = properties2D_table.style.set_table_attributes("style='display:inline'").set_caption('2D Shape Properties')

def calculate_regionprops_table(labels_tuple, image_tuple = (None, None, None),
                                properties = ["label"], dimensionalized = False,
                                exclude_boundary_labels = True):
    data_labels, kwargs_labels, type_labels = labels_tuple
    data_image, kwargs_image, type_image = image_tuple
        
    dims = kwargs_labels["metadata"]["dims"]
    new_dims = [dim for i,dim in enumerate(dims) if data_labels.shape[i]>1]
    
    data_labels, kwargs_labels = reorder_dimensions(data_labels, dims, new_dims, kwargs_labels)
    if data_image is not None:
        data_image, kwargs_image = reorder_dimensions(data_image, dims, new_dims, kwargs_image)
    if exclude_boundary_labels:
        data_labels = remove_boundary_labels(data_labels)
    
    dims = kwargs_labels["dims"]
    units = kwargs_labels["units"]
    spatial_dims = [*dims[1:]]
    sdim = len(spatial_dims)
    
    timelapse = False
    if "t" in dims:
        timelapse = True
        
    # additional properties
    if "label" not in properties:
        properties = properties + ["label"]
        
    subsequent_properties = []
    if "aspect_ratio" in properties:
        idx = properties.index("aspect_ratio")
        subsequent_properties.append(properties.pop(idx))
        if "major_axis_length" not in properties:
            properties = properties + ["major_axis_length"]
        if "minor_axis_length" not in properties:
            properties = properties + ["minor_axis_length"]
    
    if "roundness" in properties:
        idx = properties.index("roundness")
        subsequent_properties.append(properties.pop(idx))
        if "area" not in properties:
            properties = properties + ["area"]
        if "major_axis_length" not in properties:
            properties = properties + ["major_axis_length"]
            
    if "circularity" in properties:
        idx = properties.index("circularity")
        subsequent_properties.append(properties.pop(idx))
        if "area" not in properties:
            properties = properties + ["area"]
        if "perimeter" not in properties:
            properties = properties + ["perimeter"]
    
    if "perimeter_area_ratio" in properties:
        idx = properties.index("perimeter_area_ratio")
        subsequent_properties.append(properties.pop(idx))
        if "area" not in properties:
            properties = properties + ["area"]
        if "perimeter" not in properties:
            properties = properties + ["perimeter"]
    
    # calculate regionprops
    if timelapse:
        nframes = len(data_labels)
        if data_image is None:
            data_image = [None]*nframes
        df_list = []
        for frame in range(nframes):
            props = measure.regionprops_table(data_labels[frame], data_image[frame], spacing = kwargs_labels["scale"][1:], properties=properties)
            props["FrameNumber"] = np.array([frame]*len(props["label"]))
            df_list.append(pd.DataFrame(props))

        df = pd.concat(df_list, ignore_index = True)
    else:
        props = measure.regionprops_table(data_labels, data_image, spacing = kwargs_labels["scale"], properties=properties)
        df = pd.DataFrame(props)
    
    #if group_label:
    df["GroupLabel"] = [kwargs_labels["name"]]*len(df)
    
    # calculate subsequent properties
    if "aspect_ratio" in subsequent_properties:
        df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']
    if "roundness" in subsequent_properties:
        df['roundness'] = 4 * df['area'] / np.pi / df['major_axis_length']**2
    if "circularity" in subsequent_properties:
        df['circularity'] = 4 * np.pi * df['area'] / df['perimeter']**2
    if "perimeter_area_ratio" in subsequent_properties:
        df["perimeter_area_ratio"] = df["perimeter"] / (df["area"]**(1/sdim))
    
    # relabel dimensions in df
    df_dim_labels = ['-0', '-1', '-2'][:sdim]
    new_dim_labels = ["_"+dim for dim in spatial_dims]

    keys_dict = {}
    for key in df.keys():
        if "moments" in key or "inertia" in key:
            continue
        for i, dim_label in enumerate(df_dim_labels):
            if dim_label in key:
                new_key = key.split(dim_label)[0]
                new_key = new_key + new_dim_labels[i]
                keys_dict[key] = new_key

    df = df.rename(keys_dict, axis = 1)
    
    # dimensionalize df if requested
    if dimensionalized:
        length_unit = units[-sdim]
        keys_to_dimensionalize = {}
        for key in df.keys():
            if key in properites_units:
                unit = properites_units[key]
                unit = unit.replace("length_unit", length_unit)
                unit = unit.replace("sdim",str(sdim))
                keys_to_dimensionalize[key] = unit
        #print(keys_to_dimensionalize)
        df = df.astype(keys_to_dimensionalize)
    
    return df

def remove_boundary_labels(data_labels):
    data_new = np.array(data_labels)
    fns = data_labels.shape[0]
    for fn in range(fns):
        data_frame = data_new[fn]
        sdim = len(data_frame.shape)
        boundary_values = [data_frame.take([0,-1],axis = axis).flatten() for axis in range(sdim)]
        #left, right = data_frame[:,0], data_frame[:,-1]
        #top, bottom = data_frame[0,:], data_frame[-1,:]
        labels_boundary = np.unique(np.concatenate(boundary_values))
        for label in labels_boundary:
            if label == 0:
                continue
            data_frame[data_frame == label] = 0
    return data_new


def drop_units(df, return_units = False):
    df = df.pint.dequantify()
    
    if return_units:
        units = dict(df.columns)
    
    df.columns = df.columns.droplevel("unit")
    df = df.reset_index(drop = True)
    
    if return_units:
        return df, units
    return df

def lineplot_over_time(data, attr_name, ax = None, return_figure = False, show_figure = True, **kwargs):
    
    if not show_figure:
        plt.ioff()
    else:
        plt.ion()
    
    if not ax:
        fig, ax = plt.subplots()
    
    df, units = drop_units(data, return_units = True)
    
    lp = sns.lineplot(data=df, x="FrameNumber", y=attr_name, ax = ax, **kwargs)
    
    if units[attr_name]:
        unit_str = " [" + str(units[attr_name]) + "]"
    else:
        unit_str = ""
    ax.set_ylabel(attr_name + unit_str)
    ax.set_title(attr_name + " over time")
    
    if return_figure:
        return lp.figure
    
def statsannotator_setup(data, x, y, ax, stats_dict, order = None):
    stats_dict = copy.deepcopy(stats_dict)
    pairs = stats_dict.pop("pairs")
    annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
    annotator.configure(**stats_dict)
    return annotator

def boxplot(data, attr_name, group_by = None, ax = None, return_figure = False, show_figure = True, stats_dict = {}, **kwargs):
    
    if not show_figure:
        plt.ioff()
        
    else:
        plt.ion()
    
    if not ax:
        fig, ax = plt.subplots()
    
        
    df, units = drop_units(data, return_units = True)

    bp = sns.boxplot(data = df, x=group_by, y=attr_name, ax = ax, showmeans = True, **kwargs)
    
    if units[attr_name]:
        unit_str = " [" + str(units[attr_name]) + "]"
    else:
        unit_str = ""
    ax.set_ylabel(attr_name + unit_str)
    
    
    if len(stats_dict) > 0:
        annotator = statsannotator_setup(data = df, x=group_by, y=attr_name, ax = ax, stats_dict = stats_dict, order = kwargs.get("order"))
        annotator.apply_and_annotate()
    
    if return_figure:
        return bp.figure