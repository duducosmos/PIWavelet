from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scipy.ndimage import zoom


def resize_scalogram(
    image: NDArray,
    *,
    size: tuple[int, int],
    order: int = 1,
) -> NDArray:
    """
    Resize a scalogram image.

    Parameters
    ----------
    image
        Input 2D image.

    size
        Target size:

            (height, width)

    order
        Interpolation order.

        0 = nearest
        1 = bilinear
        3 = cubic

    Returns
    -------
    np.ndarray
    """

    image = np.asarray(image)

    if image.ndim != 2:

        raise ValueError(
            "image must be 2D"
        )

    target_h, target_w = size

    scale_h = (
        target_h
        / image.shape[0]
    )

    scale_w = (
        target_w
        / image.shape[1]
    )

    resized = zoom(
        image,
        zoom=(scale_h, scale_w),
        order=order,
    )

    return resized


def to_uint8(
    image: NDArray,
) -> NDArray[np.uint8]:
    """
    Convert normalized image to uint8.
    """

    image = np.asarray(
        image,
        dtype=np.float64,
    )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return (
        image * 255.0
    ).astype(np.uint8)


def stack_channels(
    *channels: NDArray,
) -> NDArray:
    """
    Stack multiple 2D channels into
    an image tensor.

    Returns
    -------
    np.ndarray

        Shape:

            (channels, height, width)
    """

    arrays = [
        np.asarray(ch)
        for ch in channels
    ]

    shape = arrays[0].shape

    for arr in arrays:

        if arr.shape != shape:

            raise ValueError(
                "all channels must "
                "have same shape"
            )

    return np.stack(
        arrays,
        axis=0,
    )


def to_rgb(
    r: NDArray,
    g: NDArray,
    b: NDArray,
) -> NDArray:
    """
    Build RGB image from three channels.

    Returns
    -------
    np.ndarray

        Shape:

            (height, width, 3)
    """

    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)

    if (
        r.shape != g.shape
        or r.shape != b.shape
    ):

        raise ValueError(
            "RGB channels must "
            "have same shape"
        )

    return np.stack(
        [r, g, b],
        axis=-1,
    )
