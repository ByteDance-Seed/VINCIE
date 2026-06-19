from math import ceil
from typing import Tuple


def get_window_op(name: str):
    if name == "win":
        return make_windows
    if name == "win_by_size":
        return by_size(make_windows)
    if name == "swin":
        return make_shifted_windows
    if name == "swin_by_size":
        return by_size(make_shifted_windows)
    if name == "dwin":
        return make_dilated_windows
    if name == "dwin_by_size":
        return by_size(make_dilated_windows)
    raise ValueError(f"Unknown windowing method: {name}")


# -------------------------------- Windowing -------------------------------- #


def make_windows(
    size: Tuple[int, int, int], num_windows: Tuple[int, int, int], skip_empty: bool = True
):
    t, h, w = size
    nt, nh, nw = num_windows
    wt, wh, ww = ceil(t / nt), ceil(h / nh), ceil(w / nw)  # window size.
    # NB: for loop in order of `t h w` is more cpu addressing friendly than `w h t`.
    return [
        (
            slice(it * wt, min((it + 1) * wt, t)),
            slice(ih * wh, min((ih + 1) * wh, h)),
            slice(iw * ww, min((iw + 1) * ww, w)),
        )
        for it in range(nt)
        if not skip_empty or min((it + 1) * wt, t) > it * wt
        for ih in range(nh)
        if not skip_empty or min((ih + 1) * wh, h) > ih * wh
        for iw in range(nw)
        if not skip_empty or min((iw + 1) * ww, w) > iw * ww
    ]


def make_shifted_windows(
    size: Tuple[int, int, int], num_windows: Tuple[int, int, int], skip_empty: bool = True
):
    t, h, w = size
    nt, nh, nw = num_windows
    wt, wh, ww = ceil(t / nt), ceil(h / nh), ceil(w / nw)  # window size.
    nt, nh, nw = (  # number of window.
        nt + 1 if nt > 1 else nt,
        nh + 1 if nh > 1 else nh,
        nw + 1 if nw > 1 else nw,
    )
    st, sh, sw = (  # shift size.
        0.5 if nt > 1 else 0,
        0.5 if nh > 1 else 0,
        0.5 if nw > 1 else 0,
    )
    return [
        (
            slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
            slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
            slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)),
        )
        for it in range(nt)
        if not skip_empty or min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
        for ih in range(nh)
        if not skip_empty or min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
        for iw in range(nw)
        if not skip_empty or min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
    ]


def make_dilated_windows(
    size: Tuple[int, int, int], num_windows: Tuple[int, int, int], skip_empty: bool = True
):
    t, h, w = size
    nt, nh, nw = num_windows
    return [
        (
            slice(it, t, nt),
            slice(ih, h, nh),
            slice(iw, w, nw),
        )
        for it in range(nt)
        if not skip_empty or t > it
        for ih in range(nh)
        if not skip_empty or h > ih
        for iw in range(nw)
        if not skip_empty or w > iw
    ]


# -------------------------------- Conversion ------------------------------- #


def by_size(fn):
    def _by_size(
        size: Tuple[int, int, int],
        window_size: Tuple[int, int, int],
        skip_empty: bool = True,
    ):
        t, h, w = size
        wt, wh, ww = window_size
        wt, wh, ww = (  # if 0, means no split
            wt if wt > 0 else t,
            wh if wh > 0 else h,
            ww if ww > 0 else w,
        )
        nt, nh, nw = ceil(t / wt), ceil(h / wh), ceil(w / ww)
        return fn((t, h, w), (nt, nh, nw), skip_empty)

    return _by_size
