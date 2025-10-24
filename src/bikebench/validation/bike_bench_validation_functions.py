from typing import List
import torch
import math
from bikebench.validation.base_validation_function import ValidationFunction, FeatureStore


POSITIVE_COLS = ['CS textfield', 'Stack', 'Head angle',
       'Head tube length textfield', 'Seat stay junction0', 'Seat tube length',
       'Seat angle', 'DT Length', 'FORK0R', 'BB diameter', 'ttd', 'dtd', 'csd',
       'ssd', 'Chain stay position on BB', 'SSTopZOFFSET',
       'Head tube upper extension2', 'Seat tube extension2',
       'Head tube lower extension2', 'SEATSTAYbrdgshift', 'CHAINSTAYbrdgshift',
       'SEATSTAYbrdgdia1', 'CHAINSTAYbrdgdia1', 'Dropout spacing',
       'Wall thickness Bottom Bracket', 'Wall thickness Top tube',
       'Wall thickness Head tube', 'Wall thickness Down tube',
       'Wall thickness Chain stay', 'Wall thickness Seat stay',
       'Wall thickness Seat tube', 'Wheel diameter front', 'RDBSD',
       'Wheel diameter rear', 'FDBSD', 'BB length',
       'htd', 'Wheel cut', 'std', 'Number of cogs',
       'Number of chainrings', 'FIRST color R_RGB',
       'FIRST color G_RGB', 'FIRST color B_RGB', 'SPOKES composite front',
       'SPOKES composite rear', 'SBLADEW front', 'SBLADEW rear', 'Seatpost LENGTH']

ZERO_IS_VALID_COLS = ['FIRST color R_RGB',
       'FIRST color G_RGB', 'FIRST color B_RGB']

class SaddleHeightTooSmall(ValidationFunction):
    def friendly_name(self) -> str: return "Saddle Height Too Small"
    def variable_names(self) -> List[str]: return ["Saddle height"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return 100.0 - ctx.col("Saddle height")


class SaddleCollidesWithSeatTube(ValidationFunction):
    def friendly_name(self) -> str: return "Saddle Collides With Seat Tube"
    def variable_names(self) -> List[str]: return ["Saddle height", "Seat tube length"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return ctx.col("Seat tube length") + 40.0 - ctx.col("Saddle height")


class SaddleTooShort(ValidationFunction):
    def friendly_name(self) -> str: return "Saddle Too Short"
    def variable_names(self) -> List[str]: return ["Saddle length"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return 228.0 - ctx.col("Saddle length")


class HeadAngleOverLimit(ValidationFunction):
    def friendly_name(self) -> str: return "Head Angle Over Limit"
    def variable_names(self) -> List[str]: return ["Head angle"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return ctx.col("Head angle") - 180.0


class SeatAngleOverLimit(ValidationFunction):
    def friendly_name(self) -> str: return "Seat Angle Over Limit"
    def variable_names(self) -> List[str]: return ["Seat angle"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return ctx.col("Seat angle") - 180.0


class SeatPostTooShort(ValidationFunction):
    def friendly_name(self) -> str: return "Seat Post Too Short"
    def variable_names(self) -> List[str]:
        return ["Seat tube length", "Seatpost LENGTH", "Saddle height", "Seat angle"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        theta_st = ctx.theta_st
        thresh = 55.0 * torch.sin(theta_st) - 10.0 * torch.cos(theta_st)
        buffer = 10.0  # 10 mm overlap
        return ctx.col("Saddle height") - (
            ctx.col("Seat tube length") + ctx.col("Seatpost LENGTH") + thresh - buffer
        )


class SeatPostTooLong(ValidationFunction):
    def friendly_name(self) -> str: return "Seat Post Too Long"
    def variable_names(self) -> List[str]:
        return ["Seatpost LENGTH", "Saddle height", "Seat angle"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        theta_st = ctx.theta_st
        thresh = 55.0 * torch.sin(theta_st) - 10.0 * torch.cos(theta_st)
        return thresh + ctx.col("Seatpost LENGTH") - ctx.col("Saddle height")


class RearWheelInnerDiameterTooSmall(ValidationFunction):
    def friendly_name(self) -> str: return "Rear Wheel Inner Diameter Too Small"
    def variable_names(self) -> List[str]: return ["Wheel diameter rear", "RDBSD"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        inner_d = ctx.col("Wheel diameter rear") - 2.0 * ctx.col("RDBSD")
        return 140.0 - inner_d


class FrontWheelInnerDiameterTooSmall(ValidationFunction):
    def friendly_name(self) -> str: return "Front Wheel Inner Diameter Too Small"
    def variable_names(self) -> List[str]: return ["Wheel diameter front", "FDBSD"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        inner_d = ctx.col("Wheel diameter front") - 2.0 * ctx.col("FDBSD")
        return 140.0 - inner_d


class SeatTubeExtensionLongerThanSeatTube(ValidationFunction):
    def friendly_name(self) -> str: return "Seat Tube Extension Longer Than Seat Tube"
    def variable_names(self) -> List[str]: return ["Seat tube length", "Seat tube extension2"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return ctx.col("Seat tube extension2") - ctx.col("Seat tube length")


class HeadTubeUpperExtensionAndLowerExtensionOverlap(ValidationFunction):
    def friendly_name(self) -> str: return "Head Tube Upper Extension And Lower Extension Overlap"
    def variable_names(self) -> List[str]:
        return ["Head tube length textfield", "Head tube upper extension2", "Head tube lower extension2"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return (ctx.col("Head tube upper extension2") + ctx.col("Head tube lower extension2")) - ctx.col("Head tube length textfield")


class SeatStayJunctionLongerThanSeatTube(ValidationFunction):
    def friendly_name(self) -> str: return "Seat Stay Junction Longer Than Seat Tube"
    def variable_names(self) -> List[str]: return ["Seat tube length", "Seat stay junction0"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return ctx.col("Seat stay junction0") - ctx.col("Seat tube length")


class NonNegativeParameterIsNegative(ValidationFunction):
    def friendly_name(self) -> str: return "Non-negative Parameter Is Negative"
    def variable_names(self) -> List[str]: return POSITIVE_COLS
    def validate(self, ctx: 'FeatureStore') -> torch.Tensor:
        X = torch.stack([ctx.col(c) for c in POSITIVE_COLS], dim=1)  # (n, k)
        zero_ok = set(ZERO_IS_VALID_COLS)
        eps_vec = torch.tensor(
            [0.0 if c in zero_ok else 1e-9 for c in POSITIVE_COLS],
            dtype=X.dtype, device=X.device
        ) 
        margins = eps_vec - X
        return margins.sum(dim=1)


class ChainStaySmallerThanRearWheelRadius(ValidationFunction):
    def friendly_name(self) -> str: return "Chain Stay Smaller Than Rear Wheel Radius"
    def variable_names(self) -> List[str]: return ["CS textfield", "Wheel diameter rear"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return (ctx.col("Wheel diameter rear") * 0.5) - ctx.col("CS textfield")


class ChainStayShorterThanBBDrop(ValidationFunction):
    def friendly_name(self) -> str: return "Chain Stay Shorter Than BB Drop"
    def variable_names(self) -> List[str]: return ["CS textfield", "BB textfield"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return ctx.col("BB textfield") - ctx.col("CS textfield")


class SeatStaySmallerThanRearWheelRadius(ValidationFunction):
    def friendly_name(self) -> str: return "Seat Stay Smaller Than Rear Wheel Radius"
    def variable_names(self) -> List[str]:
        return ["CS textfield", "BB textfield","Seat tube length", "Seat stay junction0", "Seat angle", "Wheel diameter rear"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        CS   = ctx.col("CS textfield")
        BB   = ctx.col("BB textfield")
        stl  = ctx.col("Seat tube length")
        ssj0 = ctx.col("Seat stay junction0")
        theta = ctx.theta_st
        z = ctx.z_bb  # sqrt(clamp_min(..., 0.0))
        x = stl - (BB/torch.sin(theta)) - ssj0
        y = BB/torch.tan(theta)
        h = z - y
        g = torch.sqrt(torch.clamp_min(h**2 + x**2 - 2*h*x*torch.cos(theta), 0.0))
        return (ctx.col("Wheel diameter rear") * 0.5) - g


class SeatTubeCollidesWithRearWheel(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat Tube Collides With Rear Wheel"

    def variable_names(self) -> List[str]:
        return [
            "CS textfield", "BB textfield", "Seat tube length", "Seat stay junction0",
            "Seat angle", "std",
            "Seat tube type OHCLASS: 0", "Seat tube type OHCLASS: 1", "Seat tube type OHCLASS: 2",
            "Wheel cut", "Wheel diameter rear",
        ]

    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        BB        = ctx.col("BB textfield")
        stl       = ctx.col("Seat tube length")
        ssj0      = ctx.col("Seat stay junction0")
        theta     = ctx.theta_st
        z         = ctx.z_bb
        ST_OD     = ctx.col("std")
        WDR       = ctx.col("Wheel diameter rear")
        wheel_cut = ctx.col("Wheel cut")

        # continuous one-hot → aero if OHCLASS:0 is strictly the max
        st0 = ctx.col("Seat tube type OHCLASS: 0")
        st1 = ctx.col("Seat tube type OHCLASS: 1")
        st2 = ctx.col("Seat tube type OHCLASS: 2")
        aero_mask = (st0 > st1) & (st0 > st2)

        x = stl - (BB / torch.sin(theta)) - ssj0
        y = BB / torch.tan(theta)
        h = z - y
        j = h * torch.sin(theta)

        q_true  = torch.where(wheel_cut < WDR, j - 40.9, (WDR * 0.5) + ((wheel_cut - WDR) * 0.5))
        q_false = j - (ST_OD * 0.5)
        q = torch.where(aero_mask, q_true, q_false)
        return (WDR * 0.5) - q


class DownTubeCantReachHeadTube(ValidationFunction):
    def friendly_name(self) -> str: return "Down Tube Can't Reach Head Tube"
    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle", "DT Length"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return ctx.DTJY - ctx.col("DT Length")


class RearWheelCutoutSeversSeatTube(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear Wheel Cutout Severs Seat Tube"

    def variable_names(self) -> List[str]:
        return [
            "CS textfield", "BB textfield", "Seat tube length", "Seat stay junction0",
            "Seat angle",
            "Seat tube type OHCLASS: 0", "Seat tube type OHCLASS: 1", "Seat tube type OHCLASS: 2",
            "Wheel cut",
        ]

    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        BB        = ctx.col("BB textfield")
        stl       = ctx.col("Seat tube length")
        ssj0      = ctx.col("Seat stay junction0")
        theta     = ctx.theta_st
        z         = ctx.z_bb
        wheel_cut = ctx.col("Wheel cut")

        # continuous one-hot → aero if OHCLASS:0 is strictly the max
        st0 = ctx.col("Seat tube type OHCLASS: 0")
        st1 = ctx.col("Seat tube type OHCLASS: 1")
        st2 = ctx.col("Seat tube type OHCLASS: 2")
        aero_mask = (st0 > st1) & (st0 > st2)

        x = stl - (BB / torch.sin(theta)) - ssj0
        y = BB / torch.tan(theta)
        h = z - y
        j = h * torch.sin(theta)

        q_true  = j + 16.0
        q_false = torch.full_like(q_true, 1e9)  # effectively disables this path when not aero
        q = torch.where(aero_mask, q_true, q_false)
        return (wheel_cut * 0.5) - q

class FootCollidesWithFrontWheel(ValidationFunction):
    def friendly_name(self) -> str: return "Foot Collides With Front Wheel"
    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle",
                "BB textfield", "DT Length", "FORK0R", "Wheel diameter front", "Wheel diameter rear"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        FWX, FBBD = ctx.front_axle_xy()
        FCD = torch.sqrt(torch.clamp_min(FWX**2 + FBBD**2, 0.0))
        wheel_radius = ctx.col("Wheel diameter front") * 0.5
        crank_plus_foot = 268.5
        pedal_center_offset = 120.0
        return (wheel_radius**2) - (pedal_center_offset**2) - (FCD - crank_plus_foot)**2


class CrankHitsGroundInLowestPosition(ValidationFunction):
    def friendly_name(self) -> str: return "Crank Hits Ground In Lowest Position"
    def variable_names(self) -> List[str]: return ["BB textfield", "Wheel diameter rear"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return (187.5 + ctx.col("BB textfield")) - (ctx.col("Wheel diameter rear") * 0.5)


class RGBValueGreaterThan255(ValidationFunction):
    def friendly_name(self) -> str: return "RGB Value Greater Than 255"
    def variable_names(self) -> List[str]:
        return ["FIRST color R_RGB", "FIRST color G_RGB", "FIRST color B_RGB"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        X = torch.stack([ctx.col("FIRST color R_RGB"),
                         ctx.col("FIRST color G_RGB"),
                         ctx.col("FIRST color B_RGB")], dim=1)
        overflow = X - 255.0
        total = torch.clamp_min(overflow, 0.0).sum(dim=1)
        fallback = overflow.sum(dim=1)
        return torch.where(total > 0, total, fallback)


class ChainStaysIntersect(ValidationFunction):
    def friendly_name(self) -> str: return "Chain Stays Intersect"
    def variable_names(self) -> List[str]: return ["csd", "Chain stay position on BB","BB length"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        return (ctx.col("csd")*0.5 + ctx.col("Chain stay position on BB")) - (ctx.col("BB length")*0.5)


class TubeWallThicknessExceedsRadius(ValidationFunction):
    def friendly_name(self) -> str: return "Tube Wall Thickness Exceeds Radius"
    def variable_names(self) -> List[str]:
        return [
            "ttd", "Wall thickness Top tube",
            "csd", "Wall thickness Chain stay",
            "ssd", "Wall thickness Seat stay",
            "dtd", "Wall thickness Down tube",
            "htd", "Wall thickness Head tube",
            "BB diameter", "Wall thickness Bottom Bracket",
        ]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        cols = [ctx.col(n) for n in self.variable_names()]
        vals = torch.stack(cols, dim=1)
        pairs = vals.reshape(vals.shape[0], -1, 2)   # (n,7,2)
        dia = pairs[:, :, 0]
        thk = pairs[:, :, 1]
        violation = thk - dia*0.5
        return torch.sum(torch.clamp_min(violation, 0.0), dim=1)


class SeatTubeTooNarrowForSeatPost(ValidationFunction):
    def friendly_name(self) -> str: return "Seat Tube Too Narrow For Seat Post"
    def variable_names(self) -> List[str]: return ["std", "Wall thickness Seat tube"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        inner_d = ctx.col("std") - ctx.col("Wall thickness Seat tube")
        return 27.2 - inner_d


class DownTubeImproperlyJoinsHeadTube(ValidationFunction):
    def friendly_name(self) -> str: return "Down Tube Improperly Joins Head Tube"
    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle",
                "DT Length", "dtd", "htd"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        penalty_mag = 1e6
        theta_ht = ctx.theta_ht
        DTJY     = ctx.DTJY
        DTJX     = ctx.DTJX
        DT_OD    = ctx.col("dtd")
        HT_OD    = ctx.col("htd")
        htlx     = ctx.col("Head tube lower extension2")

        theta_dt = torch.atan2(DTJY, DTJX)
        relative_angle = math.pi - (theta_dt + theta_ht)

        below_0_pen  = torch.clamp_min(-relative_angle, 0.0) * penalty_mag
        above_pi_pen = torch.clamp_min(relative_angle - math.pi, 0.0) * penalty_mag
        relative_angle = torch.clamp(relative_angle, 1e-9, math.pi - 1e-9)

        L1 = DT_OD / (2.0 * torch.sin(relative_angle))
        L2 = HT_OD / (2.0 * torch.tan(relative_angle))
        return L1 + L2 - htlx + below_0_pen + above_pi_pen


class TopTubeImproperlyJoinsHeadTube(ValidationFunction):
    def friendly_name(self) -> str: return "Top Tube Improperly Joins Head Tube"
    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head tube upper extension2",
                "Seat tube extension2", "Head angle", "Seat angle", "DT Length", "ttd", "htd","Seat tube length"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        penalty_mag = 1e6
        theta_ht = ctx.theta_ht
        theta_st = ctx.theta_st
        TTJX, TTJY = ctx.TTJX, ctx.TTJY
        STJX, STJY = ctx.STJX, ctx.STJY
        TT_OD = ctx.col("ttd")
        HT_OD = ctx.col("htd")
        htux  = ctx.col("Head tube upper extension2")

        tt_dy = TTJY - STJY
        tt_dx = TTJX + STJX
        theta_tt = torch.atan2(tt_dy, tt_dx)

        relative_angle = theta_tt + theta_ht
        below_0_pen  = torch.clamp_min(-relative_angle, 0.0) * penalty_mag
        above_pi_pen = torch.clamp_min(relative_angle - math.pi, 0.0) * penalty_mag
        relative_angle = torch.clamp(relative_angle, 1e-9, math.pi - 1e-9)

        L1 = TT_OD/(2.0*torch.sin(relative_angle))
        L2 = HT_OD/(2.0*torch.tan(relative_angle))
        return L1 + L2 - htux + below_0_pen + above_pi_pen


class TopTubeImproperlyJoinsSeatTube(ValidationFunction):
    def friendly_name(self) -> str: return "Top Tube Improperly Joins Seat Tube"
    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head tube upper extension2",
                "Seat tube extension2", "Head angle", "Seat angle", "DT Length", "ttd", "std","Seat tube length"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        penalty_mag = 1e6
        theta_ht = ctx.theta_ht
        theta_st = ctx.theta_st
        TTJX, TTJY = ctx.TTJX, ctx.TTJY
        STJX, STJY = ctx.STJX, ctx.STJY
        TT_OD = ctx.col("ttd")
        ST_OD = ctx.col("std")
        stux  = ctx.col("Seat tube extension2")

        tt_dy = TTJY - STJY
        tt_dx = TTJX + STJX
        theta_tt = torch.atan2(tt_dy, tt_dx)

        relative_angle = math.pi - (theta_tt + theta_st)
        below_0_pen  = torch.clamp_min(-relative_angle, 0.0) * penalty_mag
        above_pi_pen = torch.clamp_min(relative_angle - math.pi, 0.0) * penalty_mag
        relative_angle = torch.clamp(relative_angle, 1e-9, math.pi - 1e-9)

        L1 = TT_OD/(2.0*torch.sin(relative_angle))
        L2 = ST_OD/(2.0*torch.tan(relative_angle))
        seatpost_clamp_default_offset = 12.0
        return L1 + L2 - stux + below_0_pen + above_pi_pen + seatpost_clamp_default_offset


class DownTubeCollidesWithFrontWheel(ValidationFunction):
    def friendly_name(self) -> str: return "Down Tube Collides With Front Wheel"
    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle",
                "DT Length", "BB textfield", "FORK0R", "Wheel diameter front", "Wheel diameter rear", "dtd"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        FWX, FBBD = ctx.front_axle_xy()
        DTJY, DTJX = ctx.DTJY, ctx.DTJX

        DTJ_angle = torch.atan2(DTJY, DTJX)
        FW_angle  = torch.atan2(FBBD, FWX)
        DTJBBFW_angle = DTJ_angle - FW_angle

        FW_dist = torch.sqrt(torch.clamp_min(FWX**2 + FBBD**2, 0.0))
        shortest_dist = torch.sin(DTJBBFW_angle) * FW_dist

        wheel_radius = ctx.col("Wheel diameter front") * 0.5
        tube_radius  = ctx.col("dtd") * 0.5
        return wheel_radius - (shortest_dist - tube_radius)


class SaddleHitsTopTube(ValidationFunction):
    def friendly_name(self) -> str: return "Saddle Hits Top Tube"
    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head tube upper extension2",
                "Seat tube extension2", "Head angle", "Seat angle", "DT Length", "ttd", "std",
                "Seat tube length", "Saddle height"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        theta_ht = ctx.theta_ht
        theta_st = ctx.theta_st
        TTJX, TTJY = ctx.TTJX, ctx.TTJY
        STJX, STJY = ctx.STJX, ctx.STJY
        TT_OD = ctx.col("ttd")
        SH    = ctx.col("Saddle height")

        tt_dy = TTJY - STJY
        tt_dx = TTJX + STJX
        theta_tt = torch.atan2(tt_dy, tt_dx)

        # Small-angle approx (preserved semantics)
        arc    = 21.0 - 55.0*torch.cos(theta_st) + 10.0*torch.sin(theta_st)
        radius = SH - 55.0*torch.sin(theta_st) - 10.0*torch.cos(theta_st)
        SH_angle = theta_st - arc*torch.sin(theta_st)/(radius + 1e-12)

        Saddle_Y = SH * torch.sin(SH_angle)
        Saddle_X = SH * torch.cos(SH_angle)
        Saddle_tip_X = Saddle_X - 165.0
        Saddle_tip_Y = Saddle_Y - 22.0

        Saddle_to_STJ_X = -(Saddle_tip_X - STJX)
        Saddle_to_STJ_Y =  (Saddle_tip_Y - STJY)

        Saddle_toSTJ_X_adj = Saddle_to_STJ_X - TT_OD*0.5 * torch.cos(theta_tt + math.pi/2)
        Saddle_toSTJ_Y_adj = Saddle_to_STJ_Y - TT_OD*0.5 * torch.sin(theta_tt + math.pi/2)

        Saddle_to_STJ_angle = torch.atan2(Saddle_toSTJ_Y_adj, Saddle_toSTJ_X_adj)
        return theta_tt - Saddle_to_STJ_angle


class SaddleHitsHeadTube(ValidationFunction):
    def friendly_name(self) -> str: return "Saddle Hits Head Tube"
    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head tube upper extension2",
                "Seat tube extension2", "Head angle", "Seat angle", "DT Length", "ttd", "htd",
                "Seat tube length", "Saddle height"]
    def validate(self, ctx: FeatureStore) -> torch.Tensor:
        theta_ht = ctx.theta_ht
        theta_st = ctx.theta_st
        DTJY     = ctx.DTJY
        DTJX     = ctx.DTJX
        HT_OD    = ctx.col("htd")
        SH       = ctx.col("Saddle height")
        stack    = ctx.col("Stack")

        arc    = 21.0 - 55.0*torch.cos(theta_st) + 10.0*torch.sin(theta_st)
        radius = SH - 55.0*torch.sin(theta_st) - 10.0*torch.cos(theta_st)
        SH_angle = theta_st - arc*torch.sin(theta_st)/(radius + 1e-12)

        Saddle_Y = SH * torch.sin(SH_angle)
        Saddle_X = SH * torch.cos(SH_angle)
        Saddle_tip_X = Saddle_X - 165.0
        Saddle_tip_Y = Saddle_Y - 22.0

        Saddle_to_STJ_X = (-Saddle_tip_X - DTJX)
        Saddle_to_STJ_Y = (Saddle_tip_Y - DTJY)

        mask = (Saddle_tip_Y > stack).to(Saddle_to_STJ_X.dtype)
        critical_thickness = (HT_OD*0.5) * (1.0 - mask) + 19.05 * mask

        Saddle_toSTJ_X_adj = Saddle_to_STJ_X + critical_thickness * torch.cos(theta_ht - math.pi/2)
        Saddle_toSTJ_Y_adj = Saddle_to_STJ_Y - critical_thickness * torch.sin(theta_ht - math.pi/2)

        Saddle_to_STJ_angle = torch.atan2(Saddle_toSTJ_Y_adj, -Saddle_toSTJ_X_adj)
        return Saddle_to_STJ_angle - theta_ht



bike_bench_validation_functions: List[ValidationFunction] = [
    SaddleHeightTooSmall(),
    SaddleCollidesWithSeatTube(),
    SaddleTooShort(),
    HeadAngleOverLimit(),
    SeatAngleOverLimit(),
    SeatPostTooShort(),
    SeatPostTooLong(),
    RearWheelInnerDiameterTooSmall(),
    FrontWheelInnerDiameterTooSmall(),
    SeatTubeExtensionLongerThanSeatTube(),
    HeadTubeUpperExtensionAndLowerExtensionOverlap(),
    SeatStayJunctionLongerThanSeatTube(),
    NonNegativeParameterIsNegative(),
    ChainStaySmallerThanRearWheelRadius(),
    ChainStayShorterThanBBDrop(),
    SeatStaySmallerThanRearWheelRadius(),
    SeatTubeCollidesWithRearWheel(),
    DownTubeCantReachHeadTube(),
    RearWheelCutoutSeversSeatTube(),
    FootCollidesWithFrontWheel(),
    CrankHitsGroundInLowestPosition(),
    RGBValueGreaterThan255(),
    ChainStaysIntersect(),
    TubeWallThicknessExceedsRadius(),
    SeatTubeTooNarrowForSeatPost(),
    DownTubeImproperlyJoinsHeadTube(),
    TopTubeImproperlyJoinsHeadTube(),
    TopTubeImproperlyJoinsSeatTube(),
    DownTubeCollidesWithFrontWheel(),
    SaddleHitsTopTube(),
    SaddleHitsHeadTube(),
]

difficult_validation_functions: List[ValidationFunction] = [
    # SaddleHeightTooSmall(),
    # SaddleCollidesWithSeatTube(),
    SaddleTooShort(),
    # HeadAngleOverLimit(),
    # SeatAngleOverLimit(),
    # SeatPostTooShort(),
    # SeatPostTooLong(),
    # RearWheelInnerDiameterTooSmall(),
    # FrontWheelInnerDiameterTooSmall(),
    # SeatTubeExtensionLongerThanSeatTube(),
    # HeadTubeUpperExtensionAndLowerExtensionOverlap(),
    # SeatStayJunctionLongerThanSeatTube(),
    # NonNegativeParameterIsNegative(),
    # ChainStaySmallerThanRearWheelRadius(),
    # ChainStayShorterThanBBDrop(),
    # SeatStaySmallerThanRearWheelRadius(),
    SeatTubeCollidesWithRearWheel(),
    # DownTubeCantReachHeadTube(),
    # RearWheelCutoutSeversSeatTube(),
    FootCollidesWithFrontWheel(),
    # CrankHitsGroundInLowestPosition(),
    # RGBValueGreaterThan255(),
    # ChainStaysIntersect(),
    # TubeWallThicknessExceedsRadius(),
    SeatTubeTooNarrowForSeatPost(), 
    DownTubeImproperlyJoinsHeadTube(),
    # TopTubeImproperlyJoinsHeadTube(),
    TopTubeImproperlyJoinsSeatTube(),
    # DownTubeCollidesWithFrontWheel(),
    # SaddleHitsTopTube(),
    # SaddleHitsHeadTube(),
]
