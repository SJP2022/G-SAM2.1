e | Literal["none"] | None = None,
        embed: Incomplete | None = None,
        link: Incomplete | None = None,
        noGrp: _ConvertibleToBool | None = None,
        noSelect: _ConvertibleToBool | None = None,
        noRot: _ConvertibleToBool | None = None,
        noChangeAspect: _ConvertibleToBool | None = None,
        noMove: _ConvertibleToBool | None = None,
        noResize: _ConvertibleToBool | None = None,
        noEditPoints: _ConvertibleToBool | None = None,
        noAdjustHandles: _ConvertibleToBool | None = None,
        noChangeArrowheads: _ConvertibleToBool | None = None,
        noChangeShapeType: _ConvertibleToBool | None = None,
        extLst: ExtensionList | None = None,
        alphaBiLevel: AlphaBiLevelEffect | None = None,
        alphaCeiling: AlphaCeilingEffect | None = None,
        alphaFloor: AlphaFloorEffect | None = None,
        alphaInv: AlphaInverseEffect | None = None,
        alphaMod: AlphaModulateEffect | None = None,
        alphaModFix: AlphaModulateFixedEffect | None = None,
        alphaRepl: AlphaReplaceEffect | None = None,
        biLevel: BiLevelEffect | None = None,
        blur: BlurEffect | None = None,
        clrChange: ColorChangeEffect | None = None,
        clrRepl: ColorReplaceEffect | None = None,
        duotone: DuotoneEffect | None = None,
        fillOverlay: FillOverlayEffect | None = None,
        grayscl: GrayscaleEffect | None = None,
        hsl: HSLEffect | None = None,
        lum: LuminanceEffect | None = None,
        tint: TintEffect | None = None,
    ) -> None: ...

class TileInfoProperties(Serialisable):
    tx: Integer[Literal[True]]
    ty: Integer[Literal[True]]
    sx: Integer[Literal[True]]
    sy: Integer[Literal[True]]
    flip: NoneSet[_PropertiesFlip]
    algn: Set[_TileInfoPropertiesAlgn]
    def __init__(
        self,
        tx: ConvertibleToInt | None = None,
        ty: ConvertibleToInt | None = None,
        sx: ConvertibleToInt | None = None,
        sy: ConvertibleToInt | None = None,
        flip: _PropertiesFlip | Literal["none"] | None = None,
        *,
        algn: _TileInfoPropertiesAlgn,
    ) -> None: ...

class BlipFillProperties(Serialisable):
    tagname: ClassVar[str]
    dpi: Integer[Literal[True]]
    rotWithShape: Bool[Literal[True]]
    blip: Typed[Blip, Literal[True]]
    srcRect: Typed[RelativeRect, Literal[True]]
    tile: Typed[TileInfoProperties, Literal[True]]
    stretch: Typed[StretchInfoProperties, Literal[True]]
    __elements__: ClassVar[tuple[str, ...]]
    def __init__(
        self,
        dpi: ConvertibleToInt | None = None,
        rotWithShape: _ConvertibleToBool | None = None,
        blip: Blip | None = None,
        tile: TileInfoProperties | None = None,
        stretch: StretchInfoProperties = ...,
        srcRect: RelativeRect | None = None,
    ) -> None: ...
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          from _typeshed import ConvertibleToFloat, ConvertibleToInt
from typing import ClassVar, Literal, overload
from typing_extensions import TypeAlias

from openpyxl.descriptors.base import Bool, Float, Integer, Set, String, Typed, _ConvertibleToBool
from openpyxl.descriptors.serialisable import Serialisable

from .colors import ColorChoice

_FillOverlayEffectBlend: TypeAlias = Literal["over", "mult", "screen", "darken", "lighten"]
_EffectContainerType: TypeAlias = Literal["sib", "tree"]
_Algn: TypeAlias = Literal["tl", "t", "tr", "l", "ctr", "r", "bl", "b", "br"]
_PresetShadowEffectPrst: TypeAlias = Literal[
    "shdw1",
    "shdw2",
    "shdw3",
    "shdw4",
    "shdw5",
    "shdw6",
    "shdw7",
    "shdw8",
    "shdw9",
    "shdw10",
    "shdw11",
    "shdw12",
    "shdw13",
    "shdw14",
    "shdw15",
    "shdw16",
    "shdw17",
    "shdw18",
    "shdw19",
    "shdw20",
]

class TintEffect(Serialisable):
    tagname: ClassVar[str]
    hue: Integer[Literal[False]]
    amt: Integer[Literal[False]]
    def __init__(self, hue: ConvertibleToInt = 0, amt: ConvertibleToInt = 0) -> None: ...

class LuminanceEffect(Serialisable):
    tagname: ClassVar[str]
    bright: Integer[Literal[False]]
    contrast: Integer[Literal[False]]
    def __init__(self, bright: ConvertibleToInt = 0, contrast: ConvertibleToInt = 0) -> None: ...

class HSLEffect(Serialisable):
    hue: Integer[Literal[False]]
    sat: Integer[Literal[False]]
    lum: Integer[Literal[False]]
    def __init__(self, hue: ConvertibleToInt, sat: ConvertibleToInt, lum: ConvertibleToInt) -> None: ...

class GrayscaleEffect(Serialisable):
    tagname: ClassVar[str]

class FillOverlayEffect(Serialisable):
    blend: Set[_FillOverlayEffectBlend]
    def __init__(self, blend: _FillOverlayEffectBlend) -> None: ...

class DuotoneEffect(Serialisable): ...
class ColorReplaceEffect(Serialisable): ...
class Color(Serialisable): ...

class ColorChangeEffect(Serialisable):
    useA: Bool[Literal[True]]
    clrFrom: Typed[Color, Literal[False]]
    clrTo: Typed[Color, Literal[False]]
    @overload
    def __init__(self, useA: _ConvertibleToBool | None = None, *, clrFrom: Color, clrTo: Color) -> None: ...
    @overload
    def __init__(self, useA: _ConvertibleToBool | None, clrFrom: Color, clrTo: Color) -> None: ...

class BlurEffect(Serialisable):
    rad: Float[Literal[False]]
    grow: Bool[Literal[True]]
    def __init__(self, rad: ConvertibleToFloat, grow: _ConvertibleToBool | None = None) -> None: ...

class BiLevelEffect(Serialisable):
    thresh: Integer[Literal[False]]
    def __init__(self, thresh: ConvertibleToInt) -> None: ...

class AlphaReplaceEffect(Serialisable):
    a: Integer[Literal[False]]
    def __init__(self, a: ConvertibleToInt) -> None: ...

class AlphaModulateFixedEffect(Serialisable):
    amt: Integer[Literal[False]]
    def __init__(self, amt: ConvertibleToInt) -> None: ...

class EffectContainer(Serialisable):
    type: Set[_EffectContainerType]
    name: String[Literal[True]]
    def __init__(self, type: _EffectContainerType, name: str | None = None) -> None: ...

class AlphaModulateEffect(Serialisable):
    cont: Typed[EffectContainer, Literal[False]]
    def __init__(self, cont: EffectContain