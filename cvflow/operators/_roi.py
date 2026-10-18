"""固定 ROI 裁剪：采集节点与「裁剪 ROI」节点共用的参数、裁法和坐标约定。

取图之后裁出固定尺寸的一块直接喂给深度学习节点。和"下游再裁"相比有两个实际好处：
传给后面的数据少（628×768 只有 1280×1024 的 18%），以及尺寸在源头就定死了——
换相机、换分辨率都不会悄悄改变送进网络的张量尺寸。

尺寸由 roi_w / roi_h 两个参数定，框只能拖位置（Param 的 locked_size）。每个工位一个位置，
标定一次填进去就不动了；位置可以在图像上拖，也可以直接把训练导出的 rois.json 里的数填进来。

坐标约定（全软件只有这一套，见 ``Image.origin``）：

* 裁出来的图在 ``meta["roi"] = [x, y, w, h]`` 里记着自己是**原始帧**的哪一块；连着裁两次会累加，
  始终相对最原始那帧。
* 节点画的叠加层用送进它那张图的坐标——面板上显示裁剪后的小图时，叠加层就落在小图上该在的地方。
* 节点输出的坐标（轨迹点、检测框、这里给出的 roi / offset）用原始帧坐标——PLC / 执行机构拿到的就是它。
"""
from __future__ import annotations

import numpy as np

from ..core.node import Node, NodeError, Param
from ..core.types import Image, Rect

#: ROI 框在图上的颜色（橙色，和算法结果的绿 / 红分开）
ROI_COLOR = "#ffa500"


def roi_params(w: int = 628, h: int = 768, toggle: bool = True) -> list[Param]:
    """固定 ROI 的一组参数。

    ``toggle=False`` 不带「固定 ROI 裁剪」开关——「裁剪 ROI」节点本身就是来裁的，开关没有意义；
    采集节点默认不裁，所以带开关。两边其余参数完全一样，界面上的摆框方式也一样。
    """
    ps: list[Param] = []
    if toggle:
        ps.append(Param("roi_enabled", False, "bool", label="固定 ROI 裁剪",
                        description="打开后本节点直接输出裁好的小图（尺寸就是下面的 ROI 宽/高），"
                                    "下游深度学习节点拿到的就是定尺寸输入。关掉则输出整张原图。"
                                    "裁剪位置会记在图像上，下游把坐标换算回原图时自动带上"))
    ps += [
        Param("roi_w", w, "int", min=8, max=20000, label="ROI 宽",
              description="要和模型的输入宽度一致（看导出的 sidecar json 里的 width）"),
        Param("roi_h", h, "int", min=8, max=20000, label="ROI 高",
              description="要和模型的输入高度一致（看导出的 sidecar json 里的 height）"),
        Param("roi", None, "rect", label="ROI 位置", locked_size="roi_w,roi_h",
              description="在图像上拖这个框定位置，尺寸只能用上面的宽/高改。不设就取画面正中。"
                          "多工位时每个工位各填各的（训练侧 make_rois.py 会把每个工位的位置算好）"),
    ]
    return ps


def roi_auto_enable(node: Node, name: str, value) -> None:
    """在图上摆了一个框，就是想裁——别再要求额外去勾一个开关。

    摆完框却忘了勾「固定 ROI 裁剪」，节点会原样输出整张原图，下游深度学习节点把整图缩放到
    模型输入尺寸照样能跑出结果，只是全错；现象是"显示的是原图、轨迹不在 ROI 里"，
    很难想到是少勾了一个勾。所以这里自动打开，用户要关随时能关。
    """
    if name != "roi" or not value or node.get("roi_enabled"):
        return
    r = Rect.from_dict(value) if isinstance(value, dict) else value
    if r is None or r.is_empty():
        return
    node.values["roi_enabled"] = True


def frame_rect(img: Image) -> Rect:
    """这张图对应原始帧的哪一块（原始帧坐标）。没裁过就是整幅。"""
    ox, oy = img.origin
    return Rect(float(ox), float(oy), float(img.width), float(img.height))


def fixed_roi_xy(node: Node, img: Image) -> tuple[int, int, int, int]:
    """按节点上的 ROI 宽/高和位置算出要裁的 (x, y, w, h)，在送进来这张图的坐标里。

    尺寸以参数为准、中心保持不动（改了宽/高框不用重摆）；还没摆过框就取正中，拖一下就定下来；
    框会被压回画面里（画面放得下的话）——出了画面就是在喂黑边。
    """
    w, h = int(node.get("roi_w")), int(node.get("roi_h"))
    if w <= 0 or h <= 0:
        raise NodeError("ROI 宽/高必须为正")
    r = node.rect("roi")
    if r is None or r.is_empty():
        x, y = (img.width - w) // 2, (img.height - h) // 2
    else:
        x, y = int(round(r.cx - w / 2)), int(round(r.cy - h / 2))
    x = min(max(0, x), max(0, img.width - w))
    y = min(max(0, y), max(0, img.height - h))
    return x, y, w, h


def crop_fixed(img: Image, x: int, y: int, w: int, h: int, ctx=None) -> tuple[Image, Rect]:
    """从 ``img`` 里裁出 (x, y, w, h) 这一块，输出尺寸**严格**等于 w×h。

    这是要送进网络的张量尺寸，差一个像素模型就得重导，所以框超出画面时补黑边，
    而不是像普通 ROI 那样取交集缩小。

    返回 (裁好的图, 这块图在原始帧里的矩形)。裁好的图的 ``meta["roi"]`` 记着同一个矩形，
    下游据此把坐标换算回原始帧；送进来的图若已经是裁过一次的，原点会累加上去。
    """
    if w <= 0 or h <= 0:
        raise NodeError("ROI 宽/高必须为正")
    data = img.data
    sx0, sy0 = max(0, x), max(0, y)
    sx1, sy1 = min(data.shape[1], x + w), min(data.shape[0], y + h)
    if sx1 <= sx0 or sy1 <= sy0:
        raise NodeError(f"ROI ({x},{y} {w}×{h}) 和图像（{img.width}×{img.height}）没有交集")
    out = np.zeros((h, w) + data.shape[2:], data.dtype)
    out[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = data[sy0:sy1, sx0:sx1]
    if (sx1 - sx0, sy1 - sy0) != (w, h) and ctx is not None:
        ctx.log(f"ROI ({x},{y} {w}×{h}) 超出图像 {img.width}×{img.height}，缺的部分补黑", "warning")
    ox, oy = img.origin
    rect = Rect(float(ox + x), float(oy + y), float(w), float(h))
    cropped = img.derive(out)
    cropped.meta["roi"] = [ox + x, oy + y, w, h]
    return cropped, rect


def apply_roi(node: Node, img: Image, ctx) -> tuple[Image, Rect]:
    """采集节点用：开了「固定 ROI 裁剪」就按节点上的框裁一块，没开就原样输出。

    没开裁剪时返回整幅的矩形——``roi`` 这个输出口的含义始终是"输出的图对应原始帧的哪一块"，
    下游不用分情况讨论。只有真裁了才在图上记 ``meta["roi"]``，老流程的行为一点不变。
    """
    if not node.get("roi_enabled"):
        return img, frame_rect(img)
    x, y, w, h = fixed_roi_xy(node, img)
    return crop_fixed(img, x, y, w, h, ctx)
