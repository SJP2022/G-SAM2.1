eral[False]]
    dist: Float[Literal[False]]
    dir: Integer[Literal[False]]
    # Same as parent
    # scrgbClr = ColorChoice.scrgbClr
    # srgbClr = ColorChoice.srgbClr
    # hslClr = ColorChoice.hslClr
    # sysClr = ColorChoice.sysClr
    # schemeClr = ColorChoice.schemeClr
    # prstClr = ColorChoice.prstClr
    __elements__: ClassVar[tuple[str, ...]]
    def __init__(self, blurRad: ConvertibleToFloat, dist: ConvertibleToFloat, dir: ConvertibleToInt, **kw) -> None: ...

class OuterShadow(ColorChoice):
    tagname: ClassVar[str]
    blurRad: Float[Literal[True]]
    dist: Float[Literal[True]]
    dir: Integer[Literal[True]]
    sx: Integer[Literal[True]]
    sy: Integer[Literal[True]]
    kx: Integer[Literal[True]]
    ky: Integer[Literal[True]]
    algn: Set[_Algn]
    rotWithShape: Bool[Literal[True]]
    # Same as parent
    # scrgbClr = ColorChoice.scrgbClr
    # srgbClr = ColorChoice.srgbClr
    # hslClr = ColorChoice.hslClr
    # sysClr = ColorChoice.sysClr
    # schemeClr = ColorChoice.schemeClr
    # prstClr = ColorChoice.prstClr
    __elements__: ClassVar[tuple[str, ...]]
    @overload
    def __init__(
        self,
        blurRad: ConvertibleToFloat | None = None,
        dist: ConvertibleToFloat | None = None,
        dir: ConvertibleToInt | None = None,
        sx: ConvertibleToInt | None = None,
        sy: ConvertibleToInt | None = None,
        kx: ConvertibleToInt | None = None,
        ky: ConvertibleToInt | None = None,
        *,
        algn: _Algn,
        rotWithShape: _ConvertibleToBool | None = None,
        **kw,
    ) -> None: ...
    @overload
    def __init__(
        self,
        blurRad: ConvertibleToFloat | None,
        dist: ConvertibleToFloat | None,
        dir: ConvertibleToInt | None,
        sx: ConvertibleToInt | None,
        sy: ConvertibleToInt | None,
        kx: ConvertibleToInt | None,
        ky: ConvertibleToInt | None,
        algn: _Algn,
        rotWithShape: _ConvertibleToBool | None = None,
        **kw,
    ) -> None: ...

class PresetShadowEffect(ColorChoice):
    prst: Set[_PresetShadowEffectPrst]
    dist: Float[Literal[False]]
    dir: Integer[Literal[False]]
    # Same as parent
    # scrgbClr = ColorChoice.scrgbClr
    # srgbClr = ColorChoice.srgbClr
    # hslClr = ColorChoice.hslClr
    # sysClr = ColorChoice.sysClr
    # schemeClr = ColorChoice.schemeClr
    # prstClr = ColorChoice.prstClr
    __elements__: ClassVar[tuple[str, ...]]
    def __init__(self, prst: _PresetShadowEffectPrst, dist: ConvertibleToFloat, dir: ConvertibleToInt, **kw) -> None: ...

class ReflectionEffect(Serialisable):
    blurRad: Float[Literal[False]]
    stA: Integer[Literal[False]]
    stPos: Integer[Literal[False]]
    endA: Integer[Literal[False]]
    endPos: Integer[Literal[False]]
    dist: Float[Literal[False]]
    dir: Integer[Literal[False]]
    fadeDir: Integer[Literal[False]]
    sx: Integer[Literal[False]]
    sy: Integer[Literal[False]]
    kx: Integer[Literal[False]]
    ky: Integer[Literal[False]]
    algn: Set[_Algn]
    rotWithShape: Bool[Literal[True]]
    def __init__(
        self,
        blurRad: ConvertibleToFloat,
        stA: ConvertibleToInt,
        stPos: ConvertibleToInt,
        endA: ConvertibleToInt,
        endPos: ConvertibleToInt,
        dist: ConvertibleToFloat,
        dir: ConvertibleToInt,
        fadeDir: ConvertibleToInt,
        sx: ConvertibleToInt,
        sy: ConvertibleToInt,
        kx: ConvertibleToInt,
        ky: ConvertibleToInt,
        algn: _Algn,
        rotWithShape: _ConvertibleToBool | None = None,
    ) -> None: ...

class SoftEdgesEffect(Serialisable):
    rad: Float[Literal[False]]
    def __init__(self, rad: ConvertibleToFloat) -> None: ...

class EffectList(Serialisable):
    blur: Typed[BlurEffect, Literal[True]]
    fillOverlay: Typed[FillOverlayEffect, Literal[True]]
    glow: Typed[GlowEffect, Literal[True]]
    innerShdw: Typed[InnerShadowEffect, Literal[True]]
    outerShdw: Typed[OuterShadow, Literal[True]]
    prstShdw: Typed[PresetShadowEffect, Literal[True]]
    reflection: Typed[ReflectionEffect, Literal[True]]
    softEdge: Typed[SoftEdgesEffect, Literal[True]]
    __elements__: ClassVar[tuple[str, ...]]
    def __init__(
        self,
        blur: BlurEffect | None = None,
        fillOverlay: FillOverlayEffect | None = None,
        glow: GlowEffect | None = None,
        innerShdw: InnerShadowEffect | None = None,
        outerShdw: OuterShadow | None = None,
        prstShdw: PresetShadowEffect | None = None,
        reflection: ReflectionEffect | None = None,
        softEdge: SoftEdgesEffect | None = None,
    ) -> None: ...
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               from _typeshed import Incomplete

from .spreadsheet_drawing import AbsoluteAnchor, OneCellAnchor

class Drawing:
    count: int
    name: str
    description: str
    coordinates: Incomplete
    left: int
    top: int
    resize_proportional: bool
    rotation: int
    anchortype: str
    anchorcol: int
    anchorrow: int
    def __init__(self) -> None: ...
    @property
    def width(self) -> int: ...
    @width.setter
    def width(self, w: int) -> None: ...
    @property
    def height(self) -> int: ...
    @height.setter
    def height(self, h: int) -> None: ...
    def set_dimension(self, w: int = 0, h: int = 0) -> None: ...
    @property
    def anchor(self) -> AbsoluteAnchor | OneCellAnchor: ...
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            from _typeshed import ConvertibleToInt
from typing import ClassVar, Literal, overload

from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.text import RichText
from openpyxl.descriptors import Typed
from openpyxl.descriptors.base import Alias, Bool, Integer, String, _ConvertibleToBool
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.drawing.geometry import ShapeStyle
from openpyxl.drawing.properties import NonVisualDrawingProps, NonVisualDrawingShapeProps

class Connection(Serialisable):
    id: Integer[Literal[False]]
    idx: Integer[Literal[False]]
    def __init__(self, id: ConvertibleToInt, idx: ConvertibleToInt) -> None: ...

class ConnectorLocking(Serialisable):
    extLst: Typed[ExtensionList, Literal[True]]
    def __init__(self, extLst: ExtensionList | None = None) -> None: ...

class NonVisualConnectorProperties(Serialisable):
    cxnSpLocks: Typed[ConnectorLocking, Literal[True]]
    stCxn: Typed[Connection, Literal[True]]
    endCxn: Typed[Connection, Literal[True]]
    extLst: Typed[ExtensionList, Literal[True]]
    def __init__(
        self,
        cxnSpLocks: ConnectorLocking | None = None,
        stCxn: Connection | None = None,
        endCxn: Connection | None = None,
        extLst: ExtensionList | None = None,
    ) -> None: ...

class ConnectorNonVisual(Serialisable):
    cNvPr: Typed[NonVisualDrawingProps, Literal[False]]
    cNvCxnSpPr: Typed[NonVisualConnectorProperties, Literal[False]]
    __elements__: ClassVar[tuple[str, ...]]
    def __init__(self, cNvPr: NonVisualDrawingProps, cNvCxnSpPr: NonVisualConnectorProperties) -> None: ...

class ConnectorShape(Serialisable):
    tagname: ClassVar[str]
    nvCxnSpPr: Typed[ConnectorNonVisual, Literal[False]]
    spPr: Typed[GraphicalProperties, Literal[False]]
    style: Typed[ShapeStyle, Literal[True]]
    macro: String[Literal[True]]
    fPublished: Bool[Literal[True]]
    def __init__(
        self,
        nvCxnSpPr: ConnectorNonVisual,
        spPr: GraphicalProperties,
        style: ShapeStyle | None = None,
        macro: str | None = None,
        fPublished: _ConvertibleToBool | None = None,
    ) -> None: ...

class ShapeMeta(Serialisable):
    tagname: ClassVar[str]
    cNvPr: Typed[NonVisualDrawingProps, Literal[False]]
    cNvSpPr: Typed[NonVisualDrawingShapeProps, Literal[False]]
    def __init__(self, cNvPr: NonVisualDrawingProps, cNvSpPr: NonVisualDrawingShapeProps) -> None: ...

class Shape(Serialisable):
    macro: String[Literal[True]]
    textlink: String[Literal[True]]
    fPublished: Bool[Literal[True]]
    fLocksText: Bool[Literal[True]]
    nvSpPr: Typed[ShapeMeta, Literal[True]]
    meta: Alias
    spPr: Typed[GraphicalProperties, Literal[False]]
    graphicalProperties: Alias
    style: Typed[ShapeStyle, Literal[True]]
    txBody: Typed[RichText, Literal[True]]
    @overload
    def __init__(
        self,
        macro: str | None = None,
        textlink: str | None = None,
        fPublished: _ConvertibleToBool | None = None,
        fLocksText: _ConvertibleToBool | None = None,
        nvSpPr: ShapeMeta | None = None,
        *,
        spPr: GraphicalProperties,
        style: ShapeStyle | None = None,
        txBody: RichText | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        macro: str | None,
        textlink: str | None,
        fPublished: _ConvertibleToBool | None,
        fLocksText: _ConvertibleToBool | None,
        nvSpPr: ShapeMeta | None,
        spPr: GraphicalProperties,
        style: ShapeStyle | None = None,
        txBody: RichText | None = None,
    ) -> None: ...
                                                                                                                                                                                                                                                                                                                     arget_grid_y * H_im // grid_h
        target_x_offset_b = target_grid_x * W_im // grid_w
        target_y_offset_e = (target_grid_y + 1) * H_im // grid_h
        target_x_offset_e = (target_grid_x + 1) * W_im // grid_w
        target_H_im_downsize = target_y_offset_e - target_y_offset_b
        target_W_im_downsize = target_x_offset_e - target_x_offset_b

        segment_downsize = F.resize(
            obj.segment[None, None],
            size=(target_H_im_downsize, target_W_im_downsize),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,  # antialiasing for downsizing
        )[0, 0]
        if should_hflip[target_grid_y, target_grid_x].item():
            segment_downsize = F.hflip(segment_downsize[None, None])[0, 0]

        segment_output[
            target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
        ] = segment_downsize
        obj.segment = segment_output

    return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, prob=0.15, grid_h=2, grid_w=2, use_random_hflip=False):
        self.prob = prob
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_random_hflip = use_random_hflip

    def __call__(self, datapoint, **kwargs):
        if random.random() > self.prob:
            return datapoint

        # select a random location to place the target mask in the mosaic
        target_grid_y = random.randint(0, self.grid_h - 1)
        target_grid_x = random.randint(0, self.grid_w - 1)
        # whether to flip each grid in the mosaic horizontally
        if self.use_random_hflip:
            should_hflip = torch.rand(self.grid_h, self.grid_w) < 0.5
        else:
            should_hflip = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        for i in range(len(datapoint.frames)):
            datapoint = random_mosaic_frame(
                datapoint,
                i,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                target_grid_y=target_grid_y,
                target_grid_x=target_grid_x,
                should_hflip=should_hflip,
            )

        return datapoint
